from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import base64
import io
from unittest.mock import patch

import cv2
import numpy as np

from config import Config
from mddad import create_app
from mddad.model_options import supported_behavior_backends, supported_phone_backends


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProductContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()

        class TestConfig(Config):
            TESTING = True
            SECRET_KEY = "test-secret"
            DATABASE_PATH = str(Path(cls.temp_dir.name) / "test.sqlite3")
            UPLOAD_FOLDER = str(Path(cls.temp_dir.name) / "uploads")
            SEED_DEMO_USER = False
            ENABLE_LEGACY_RESEARCH_ROUTES = False

        cls.app = create_app(TestConfig)
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_only_product_backends_are_exposed(self):
        self.assertEqual(supported_behavior_backends(), {"driver_safety_v1"})
        self.assertEqual(
            supported_phone_backends(),
            {"disabled", "rtdetr_finetuned"},
        )

    def test_legacy_research_routes_are_disabled(self):
        rules = {rule.rule for rule in self.app.url_map.iter_rules()}
        self.assertNotIn("/multimodal-test", rules)
        self.assertNotIn("/mmrt-v2-demo", rules)
        self.assertNotIn("/sessions/<int:session_id>/multimodal-analysis", rules)
        self.assertIn("/realtime", rules)

    def test_health_exposes_product_pipeline(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["pipeline"], "driver_safety_v1")
        self.assertFalse(payload["legacy_research_routes"])

    def test_model_health_does_not_expose_local_paths(self):
        response = self.client.get("/health/model")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["pipeline"], "driver_safety_v1")
        self.assertNotIn("path", json.dumps(payload).lower())
        self.assertIn(payload["identity_status"], {"ready_for_enrollment", "requires_tensorflow_runtime"})

    def test_driver_table_has_face_enrollment_columns(self):
        with self.app.app_context():
            from mddad.database import get_db

            columns = {
                row["name"]
                for row in get_db().execute("SELECT name FROM pragma_table_info('drivers')").fetchall()
            }
        self.assertIn("face_embedding_json", columns)
        self.assertIn("face_enrolled_at", columns)
        self.assertIn("face_enrollment_image_path", columns)

    def test_shared_product_database_tables_exist(self):
        expected = {
            "model_artifacts",
            "media_assets",
            "driver_biometrics",
            "inference_runs",
            "model_predictions",
            "session_models",
            "app_audit_logs",
        }
        with self.app.app_context():
            from mddad.database import get_db

            tables = {
                row["name"]
                for row in get_db().execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        self.assertTrue(expected.issubset(tables))

    def test_model_artifacts_are_seeded_for_driver_safety_v1(self):
        with self.app.app_context():
            from mddad.database import get_db

            rows = get_db().execute(
                "SELECT backend_key FROM model_artifacts ORDER BY backend_key"
            ).fetchall()
        self.assertEqual(
            [row["backend_key"] for row in rows],
            [
                "driver_safety_v1.action_recognition",
                "driver_safety_v1.drowsiness",
                "driver_safety_v1.face_detection",
                "driver_safety_v1.face_verification",
                "driver_safety_v1.phone_presence",
                "driver_safety_v1.scaler",
            ],
        )

    def test_database_health_reports_shared_tables(self):
        response = self.client.get("/health/db")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("model_artifacts", payload["tables"])
        self.assertIn("inference_runs", payload["tables"])
        self.assertEqual(len(payload["registered_models"]), 6)
        self.assertNotIn("artifact_path", json.dumps(payload["registered_models"]))

    def test_shared_repository_records_project_flow(self):
        with self.app.app_context():
            from mddad.database import get_db
            from mddad.repositories import (
                create_audit_log,
                create_inference_run,
                create_media_asset,
                create_model_prediction,
                create_session_model,
                finish_inference_run,
                get_model_artifact_id,
                upsert_driver_biometric,
            )

            db = get_db()
            user_id = db.execute(
                """
                INSERT INTO users (username, password_hash, display_name)
                VALUES ('tester', 'hash', 'Tester')
                """
            ).lastrowid
            driver_id = db.execute(
                """
                INSERT INTO drivers (user_id, display_name)
                VALUES (?, 'Driver Test')
                """,
                (user_id,),
            ).lastrowid
            session_id = db.execute(
                """
                INSERT INTO sessions (user_id, driver_id, session_type, status)
                VALUES (?, ?, 'uploaded_video', 'created')
                """,
                (user_id, driver_id),
            ).lastrowid

            media_id = create_media_asset(
                user_id=user_id,
                driver_id=driver_id,
                session_id=session_id,
                asset_type="uploaded_video",
                storage_path="uploads/test.mp4",
                metadata={"source": "test"},
            )
            model_id = get_model_artifact_id("driver_safety_v1.drowsiness")
            run_started = 0.0
            run_id = create_inference_run(
                user_id=user_id,
                driver_id=driver_id,
                session_id=session_id,
                run_type="uploaded_video_analysis",
                model_backend="driver_safety_v1.drowsiness",
                model_artifact_id=model_id,
                input_asset_id=media_id,
            )
            create_session_model(
                session_id=session_id,
                model_backend="driver_safety_v1.drowsiness",
                model_role="uploaded_behavior_analysis",
                model_artifact_id=model_id,
                threshold=0.5,
            )
            create_model_prediction(
                run_id=run_id,
                session_id=session_id,
                model_backend="driver_safety_v1.drowsiness",
                timestamp_seconds=1.5,
                frame_index=12,
                label="drowsy",
                confidence=0.81,
                probabilities={"drowsy": 0.81, "normal": 0.19},
                features={"ear": 0.17},
            )
            biometric_id = upsert_driver_biometric(
                user_id=user_id,
                driver_id=driver_id,
                biometric_type="face_embedding",
                model_backend="driver_safety_v1.face_verification",
                embedding_json="[0.0,1.0]",
                enrollment_asset_id=media_id,
                threshold=0.94,
            )
            finish_inference_run(
                run_id,
                status="completed",
                started_monotonic=run_started,
                summary={"events": 1},
            )
            create_audit_log(
                user_id=user_id,
                driver_id=driver_id,
                session_id=session_id,
                action="test.flow_recorded",
                entity_type="inference_run",
                entity_id=run_id,
            )
            db.commit()

            self.assertGreater(media_id, 0)
            self.assertGreater(run_id, 0)
            self.assertGreater(biometric_id, 0)
            self.assertEqual(
                db.execute("SELECT COUNT(*) AS count FROM model_predictions").fetchone()["count"],
                1,
            )
            self.assertEqual(
                db.execute("SELECT status FROM inference_runs WHERE id = ?", (run_id,)).fetchone()["status"],
                "completed",
            )

    def test_face_enrollment_embedding_average_is_normalized(self):
        from mddad.face_identity import average_embeddings, embedding_consistency

        first = np.zeros(128, dtype=np.float32)
        first[0] = 1.0
        second = np.zeros(128, dtype=np.float32)
        second[0] = 0.9
        second[1] = 0.1
        second = second / np.linalg.norm(second)

        averaged = average_embeddings([first, second])
        quality = embedding_consistency([first, second])

        self.assertEqual(averaged.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(averaged)), 1.0, places=5)
        self.assertEqual(quality["sample_count"], 2)
        self.assertIsNotNone(quality["mean_distance_to_centroid"])

    def test_camera_face_enrollment_payload_is_decoded(self):
        from mddad.driver_routes import _decode_captured_face_samples

        encoded = base64.b64encode(b"fake-jpeg-bytes").decode("ascii")
        payload = json.dumps(
            [
                {
                    "data_url": f"data:image/jpeg;base64,{encoded}",
                    "pose_hint": "Look straight",
                }
            ]
        )
        samples = _decode_captured_face_samples(payload)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["filename"], "camera_sample_01.jpg")
        self.assertEqual(samples[0]["mime_type"], "image/jpeg")
        self.assertEqual(samples[0]["bytes"], b"fake-jpeg-bytes")
        self.assertEqual(samples[0]["source"], "camera_capture")

    def test_face_enrollment_route_accepts_camera_samples(self):
        from mddad.database import get_db

        with self.app.app_context():
            db = get_db()
            user_id = db.execute(
                """
                INSERT INTO users (username, password_hash, display_name)
                VALUES ('enroll-camera-user', 'hash', 'Enroll Camera User')
                """
            ).lastrowid
            driver_id = db.execute(
                """
                INSERT INTO drivers (user_id, display_name)
                VALUES (?, 'Camera Route Driver')
                """,
                (user_id,),
            ).lastrowid
            db.commit()

        with self.client.session_transaction() as session_data:
            session_data["user_id"] = user_id
            session_data["language"] = "en"

        image = np.full((240, 240, 3), 220, dtype=np.uint8)
        ok, buffer = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")

        class FakeVerifier:
            ready = True
            error = None

            def embed_frame(self, frame, face_box):
                return np.concatenate([np.array([1.0], dtype=np.float32), np.zeros(127, dtype=np.float32)])

        with patch("mddad.driver_routes.get_face_identity_verifier", return_value=FakeVerifier()), patch(
            "mddad.driver_routes._detect_largest_face",
            return_value=(40, 40, 180, 180),
        ):
            response = self.client.post(
                f"/drivers/{driver_id}/face-enrollment",
                data={
                    "face_samples": (io.BytesIO(buffer.tobytes()), "camera_sample_01.jpg"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 302)
        with self.app.app_context():
            row = get_db().execute(
                """
                SELECT face_embedding_json, face_enrolled_at
                FROM drivers
                WHERE id = ?
                """,
                (driver_id,),
            ).fetchone()
            biometric = get_db().execute(
                """
                SELECT embedding_json, status
                FROM driver_biometrics
                WHERE driver_id = ?
                """,
                (driver_id,),
            ).fetchone()
        self.assertIsNotNone(row["face_embedding_json"])
        self.assertIsNotNone(row["face_enrolled_at"])
        self.assertIsNotNone(biometric["embedding_json"])
        self.assertEqual(biometric["status"], "active")

    def test_manifest_matches_temporal_contract(self):
        manifest = json.loads(
            (PROJECT_ROOT / "models" / "driver_safety_v1" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        contract = manifest["feature_contract"]
        self.assertEqual(contract["sequence_length"], 15)
        self.assertEqual(contract["feature_count"], 6)
        self.assertEqual(
            contract["feature_order"],
            [
                "left_ear",
                "right_ear",
                "mar",
                "pitch_degrees",
                "yaw_degrees",
                "roll_degrees",
            ],
        )

    def test_openvino_action_model_contract(self):
        from mddad.driver_action_openvino import ACTION_LABELS, OPENVINO_LABELS, action_model_readiness

        with self.app.app_context():
            readiness = action_model_readiness(self.app)

        self.assertTrue(readiness["available"])
        self.assertEqual(readiness["window_size"], 16)
        # The OpenVINO decoder emits nine raw classes. The product contract
        # merges left/right variants before exposing labels to the UI.
        self.assertEqual(len(OPENVINO_LABELS), 7)
        self.assertEqual(len(ACTION_LABELS), 9)
        self.assertIn("normal_driving", OPENVINO_LABELS)
        self.assertIn("phone_calling", OPENVINO_LABELS)
        self.assertIn("phone_calling", ACTION_LABELS)
        self.assertIn("phone_calling", ACTION_LABELS)

    def test_realtime_page_declares_phone_backend_explicitly(self):
        from mddad.database import get_db

        with self.app.app_context():
            db = get_db()
            user_id = db.execute(
                "INSERT INTO users (username, password_hash, display_name) VALUES ('realtime-user', 'hash', 'Realtime User')"
            ).lastrowid
            driver_id = db.execute(
                "INSERT INTO drivers (user_id, display_name) VALUES (?, 'Realtime Driver')",
                (user_id,),
            ).lastrowid
            db.commit()

        with self.client.session_transaction() as session_data:
            session_data["user_id"] = user_id
            session_data["active_driver_id"] = driver_id
            session_data["language"] = "vi"

        response = self.client.get("/realtime")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('data-phone-backend="disabled"', html)
        self.assertIn("phone", html.lower())

    def test_realtime_socket_forwards_explicit_model_contract(self):
        from mddad import socketio
        from mddad.database import get_db

        with self.app.app_context():
            db = get_db()
            user_id = db.execute(
                "INSERT INTO users (username, password_hash, display_name) VALUES ('socket-user', 'hash', 'Socket User')"
            ).lastrowid
            driver_id = db.execute(
                "INSERT INTO drivers (user_id, display_name) VALUES (?, 'Socket Driver')",
                (user_id,),
            ).lastrowid
            db.commit()

        with self.client.session_transaction() as session_data:
            session_data["user_id"] = user_id
            session_data["active_driver_id"] = driver_id
            session_data["language"] = "vi"

        with patch("mddad.sockets.realtime_manager.start", return_value=(True, None)) as start:
            socket_client = socketio.test_client(self.app, flask_test_client=self.client)
            self.assertTrue(socket_client.is_connected())
            socket_client.emit(
                "start_realtime_session",
                {
                    "behavior_model_backend": "driver_safety_v1",
                    "phone_detector_backend": "rtdetr_finetuned",
                    "language": "vi",
                },
            )
            received = socket_client.get_received()
            socket_client.disconnect()

        self.assertTrue(any(item["name"] == "realtime_status" for item in received))
        options = start.call_args.args[-1]
        self.assertEqual(options["behavior_model_backend"], "driver_safety_v1")
        self.assertEqual(options["phone_detector_backend"], "rtdetr_finetuned")
        self.assertEqual(options["language"], "vi")


if __name__ == "__main__":
    unittest.main()
