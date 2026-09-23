"""Testes determinísticos sem downloads; testes reais são prepare_examples.py."""
import unittest
import numpy as np
from PIL import Image
from vision import apply_occlusion, box_iou, person_reid_challenges


class GeometryTest(unittest.TestCase):
    def test_iou_overlap_disjoint_degenerate(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1/3)
        self.assertEqual(box_iou([0, 0, 1, 1], [3, 3, 4, 4]), 0)
        self.assertEqual(box_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0)

    def test_mask_exact_area_and_zero_invariance(self):
        original = Image.new("RGB", (100, 100), "white")
        self.assertTrue(np.array_equal(original, apply_occlusion(original, [20, 20, 80, 80], 0)))
        output = np.array(apply_occlusion(original, [20, 20, 80, 80], .5))
        altered = np.any(output != 255, axis=2)
        self.assertEqual(altered.sum(), 30*60)
        self.assertFalse(altered[:20].any())
        self.assertRaises(ValueError, apply_occlusion, original, [0, 0, 10, 10], 2)

    def test_person_reid_challenges_have_positive_variants_and_distractors(self):
        query = Image.new("RGB", (80, 160), "white")
        distractors = [Image.new("RGB", (75, 150), "blue"), Image.new("RGB", (70, 140), "green")]
        result = person_reid_challenges(query, distractors, rotation=20, occlusion=.4, brightness=.6)
        self.assertEqual(len(result["gallery"]), 6)
        self.assertEqual(result["names"][1], "Mesma pessoa · rotação 20°")
        self.assertEqual(result["relations"][-1], "Pessoa diferente na mesma cena")


if __name__ == "__main__":
    unittest.main()
