"""Testes UI de verdade: navegação, inferência e parâmetros, sem mocks."""
from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

class AppSmokeTests(unittest.TestCase):
    def test_pages_and_real_experiments(self):
        app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=180).run()
        self.assertFalse(app.exception)
        app.switch_page("app_pages/detection.py").run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertGreater(len(app.session_state["detection_result"]["result"]["detections"]),0)
        app.switch_page("app_pages/tracking.py").run()
        self.assertFalse(app.exception)
        app.slider(key="track_frames").set_value(30)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertGreater(app.session_state["tracking_result"]["result"]["stats"]["unique_ids"],0)
        app.switch_page("app_pages/counting.py").run()
        self.assertFalse(app.exception)
        app.switch_page("app_pages/reid.py").run()
        self.assertFalse(app.exception)
        app.button(key="reid_execute").click().run()
        self.assertFalse(app.exception)
        self.assertGreater(len(app.session_state["reid_result"]["df"]),1)
        app.switch_page("app_pages/person_reid.py").run()
        self.assertFalse(app.exception)
        app.button(key="FormSubmitter:person_reid_controls-Calcular ranking de pessoas").click().run()
        self.assertFalse(app.exception)
        self.assertGreater(len(app.session_state["person_reid_result"]["table"]), 4)
        app.switch_page("app_pages/occlusion.py").run()
        self.assertFalse(app.exception)
        app.button(key="occ_execute").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["occlusion_result"]["result"]["metrics"]),5)
        app.switch_page("app_pages/research.py").run()
        self.assertFalse(app.exception)
        for value in ["Geometria","Arquitetura","Produtos"]:
            app.session_state["research_view"]=value
            app.run()
            self.assertFalse(app.exception)
        app.switch_page("app_pages/applications.py").run()
        self.assertFalse(app.exception)
        app.switch_page("app_pages/guide.py").run()
        self.assertFalse(app.exception)

if __name__=="__main__":
    unittest.main()
