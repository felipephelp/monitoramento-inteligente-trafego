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
        app.switch_page("app_pages/vlm.py").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Prompt que será enviado" in item.value for item in app.subheader))
        app.session_state["vlm_result"] = {
            "caption_en": "a busy urban road",
            "caption_pt": "uma via urbana movimentada",
            "prompt_mode": "Cena de tráfego",
            "prompt_sent": "a traffic monitoring image showing",
        }
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Saída final em português" in item.value for item in app.subheader))
        app.session_state["vlm_view"]="Fusão: câmera e contexto"
        app.session_state["fusion_vlm_result"] = {
            "visual_evidence_en": "several cars on a road",
            "visual_evidence_pt": "vários carros em uma via",
            "prompt_pt": "Evidência visual: vários carros; radar: 42 km/h.",
            "generated_pt": "Há vários carros na via e o radar informa 42 km/h.",
            "speed": 42,
            "signal": "Vermelho",
            "weather": "Seco",
        }
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Prompt preparado para a fusão" in item.value for item in app.subheader))
        self.assertTrue(any("Saída gerada pelo Qwen" in item.value for item in app.subheader))
        app.switch_page("app_pages/applications.py").run()
        self.assertFalse(app.exception)
        app.switch_page("app_pages/guide.py").run()
        self.assertFalse(app.exception)

if __name__=="__main__":
    unittest.main()
