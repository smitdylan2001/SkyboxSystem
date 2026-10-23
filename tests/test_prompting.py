"""Caption parsing and prompt building (no ML dependencies): python -m unittest discover -s tests"""

import os
import unittest

os.environ.setdefault("AUTH_DISABLED", "true")  # app.config validates the environment on import

from app.pipeline.prompting import (  # noqa: E402
    MAX_STYLE_WORDS,
    build_prompt,
    compose_description,
    inpaint_prompt,
    parse_caption,
)

TEMPLATE = "This is a panorama. The image shows {prompt}"


class ParseCaptionTest(unittest.TestCase):
    def test_two_labelled_lines(self):
        style, scene = parse_caption(
            "STYLE: stylized 3D animated film render with soft lighting\n"
            "SCENE: A sunny meadow with tall grass, a large oak tree and rolling green hills under a clear blue sky."
        )
        self.assertEqual(style, "stylized 3D animated film render with soft lighting")
        self.assertEqual(scene, "a sunny meadow with tall grass, a large oak tree and rolling green hills under a clear blue sky.")

    def test_markdown_labels_and_lead_ins(self):
        style, scene = parse_caption(
            "**STYLE:** It is a Cel-shaded anime illustration.\n\n**Scene:** The image shows a quiet street at dusk."
        )
        self.assertEqual(style, "cel-shaded anime illustration")
        self.assertEqual(scene, "a quiet street at dusk.")

    def test_unlabelled_answer_is_the_scene(self):
        self.assertEqual(parse_caption("A rocky beach at sunset."), ("", "a rocky beach at sunset."))

    def test_scene_continues_on_following_lines(self):
        _, scene = parse_caption("STYLE: watercolor painting\nSCENE: a harbour town\nwith boats in the water.")
        self.assertEqual(scene, "a harbour town with boats in the water.")

    def test_people_are_dropped_from_scene_and_style(self):
        style, scene = parse_caption(
            "STYLE: 3D animated render, with cartoon characters, bright colours\n"
            "SCENE: A park with old trees. Two people walk along the path. Soft evening light."
        )
        self.assertEqual(style, "3D animated render, bright colours")
        self.assertEqual(scene, "a park with old trees. Soft evening light.")

    def test_long_style_is_cut(self):
        style, _ = parse_caption("STYLE: " + " ".join(["very"] * 40) + " stylized render\nSCENE: a forest")
        self.assertEqual(len(style.split()), MAX_STYLE_WORDS)

    def test_acronyms_keep_their_case(self):
        style, scene = parse_caption("STYLE: 3D render\nSCENE: NYC skyline at night")
        self.assertEqual((style, scene), ("3D render", "NYC skyline at night"))


class PromptTest(unittest.TestCase):
    def test_style_leads_and_closes_the_prompt(self):
        description = compose_description(
            "a sunny meadow with a large oak tree.", "stylized 3D animated film render"
        )
        self.assertEqual(
            build_prompt(TEMPLATE, description),
            "This is a panorama. The image shows a stylized 3D animated film render of a sunny meadow with a "
            "large oak tree. Everything is in the same stylized 3D animated film render style.",
        )

    def test_article(self):
        self.assertTrue(compose_description("a street", "anime illustration").startswith("an anime illustration of"))
        self.assertTrue(compose_description("a street", "3D render").startswith("a 3D render of"))

    def test_without_style_matches_the_old_prompt(self):
        self.assertEqual(
            build_prompt(TEMPLATE, compose_description("A misty lake at dawn.")),
            "This is a panorama. The image shows a misty lake at dawn.",
        )

    def test_extra_prompt_is_kept(self):
        self.assertEqual(
            compose_description("a desert road", "realistic photograph", "golden hour"),
            "a realistic photograph of a desert road. golden hour. Everything is in the same realistic photograph style.",
        )

    def test_prompt_only_job_with_style(self):
        # use_caption=false: the client's prompt is the scene.
        self.assertEqual(
            compose_description("", "pixel art", "A castle on a hill."),
            "a pixel art of a castle on a hill. Everything is in the same pixel art style.",
        )

    def test_template_may_use_style(self):
        self.assertEqual(build_prompt("{style} panorama: {prompt}", "a beach.", "pixel art"), "pixel art panorama: a beach.")
        self.assertEqual(build_prompt("{style} panorama: {prompt}", "a beach."), "panorama: a beach.")

    def test_inpaint_prompt_keeps_the_style(self):
        description = compose_description("a park", "watercolor painting")
        self.assertIn("watercolor painting", inpaint_prompt(description))
        self.assertTrue(inpaint_prompt(description).endswith("nobody is there."))


if __name__ == "__main__":
    unittest.main()
