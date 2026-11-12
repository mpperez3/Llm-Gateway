"""Tests del canario de streams corruptos (token soup)."""

from __future__ import annotations

import unittest

from llamacpp_stack.cli import stream_canary as sc

SOUP = (
    "斑马ILLED阶回声lehmented不到的ilogegot和解返回列表stel\ufffd\ufffd谨防超凡agodgorinsky尾声世子琼斯"
    "moutchAndServe皇城GuidId依trag年年来过googalous.Configyson尧-multi白雪alum要不然agne rameramacci"
    "RLreakaku/Index饲-headingekanacher渉йнаdiff.sharedInstancebes at"
    "בפieodableйнаdiffสวัสดีनमस्ते안녕googalous"
) * 2

SPANISH = (
    "El modelo arranca coherente y de pronto colapsa. Estamos revisando el registro de peticiones "
    "para entender por que la generacion se corrompe a mitad de frase, sin presion de KV ni errores "
    "de GPU. La hipotesis principal es el speculative decoding con drafter externo."
)

ENGLISH_CODE = (
    "def resolve_model(name):\n    for model in catalog:\n        if model['id'] == name:\n"
    "            return model\n    raise KeyError(f'for model {name!r} (custom endpoint)')\n"
)

CHINESE_ONLY = (
    "这是一个用于测试的中文段落，内容完全正常，不包含任何异常字符。它模拟用户用中文提问时模型的中文回答，"
    "应该被检测器接受而不是误报。模型可以用中文连贯地解释问题。"
)

MIXED_QUOTE = (
    "El autor explicaba el termino japones 木漏れ日 y citaba una frase en ruso: привет, мир. "
    "Despues volvia al castellano para cerrar el parrafo con normalidad y sin ningun problema."
)


class ScriptClassificationTests(unittest.TestCase):
    def test_buckets(self) -> None:
        self.assertEqual(sc.script_of("a"), "latin")
        self.assertEqual(sc.script_of(" "), "space")
        self.assertEqual(sc.script_of("!"), "space")
        self.assertEqual(sc.script_of("á"), "latin")
        self.assertEqual(sc.script_of("斑"), "han")
        self.assertEqual(sc.script_of("あ"), "hiragana")
        self.assertEqual(sc.script_of("ア"), "katakana")
        self.assertEqual(sc.script_of("한"), "hangul")
        self.assertEqual(sc.script_of("ب"), "arabic")
        self.assertEqual(sc.script_of("א"), "hebrew")
        self.assertEqual(sc.script_of("д"), "cyrillic")
        self.assertEqual(sc.script_of("ก"), "thai")
        self.assertEqual(sc.script_of("क"), "devanagari")
        self.assertEqual(sc.script_of("\ufffd"), "replacement")


class ConfigTests(unittest.TestCase):
    def test_defaults_when_missing(self) -> None:
        cfg = sc.normalize_config(None)
        self.assertTrue(cfg["enabled"])
        self.assertTrue(cfg["enforce"])
        self.assertEqual(cfg["min_scripts"], 4)

    def test_clamps_and_idempotent(self) -> None:
        cfg = sc.normalize_config({"window_chars": 1, "min_scripts": 99, "max_latin_ratio": 5})
        self.assertEqual(cfg["window_chars"], 80)
        self.assertEqual(cfg["min_scripts"], 12)
        self.assertEqual(cfg["max_latin_ratio"], 1.0)
        self.assertEqual(sc.normalize_config(cfg), cfg)

    def test_string_booleans(self) -> None:
        self.assertFalse(sc.normalize_config({"enabled": "off"})["enabled"])
        self.assertTrue(sc.normalize_config({"enforce": "yes"})["enforce"])


class DetectorTests(unittest.TestCase):
    def test_soup_trips(self) -> None:
        det = sc.SoupDetector()
        self.assertTrue(det.feed(SOUP))
        self.assertTrue(det.tripped)
        stats = det.stats()
        self.assertGreaterEqual(float(stats["mixed_ratio"]), 0.12)

    def test_soup_trips_incrementally(self) -> None:
        det = sc.SoupDetector()
        tripped = False
        for i in range(0, len(SOUP), 16):
            if det.feed(SOUP[i:i + 16]):
                tripped = True
                break
        self.assertTrue(tripped)
        self.assertLess(det.chars_seen, len(SOUP))

    def test_normal_spanish_does_not_trip(self) -> None:
        det = sc.SoupDetector()
        for i in range(0, len(SPANISH), 24):
            self.assertFalse(det.feed(SPANISH[i:i + 24]))

    def test_code_does_not_trip(self) -> None:
        det = sc.SoupDetector()
        for i in range(0, len(ENGLISH_CODE), 24):
            self.assertFalse(det.feed(ENGLISH_CODE[i:i + 24]))

    def test_chinese_only_does_not_trip(self) -> None:
        det = sc.SoupDetector()
        for i in range(0, len(CHINESE_ONLY), 24):
            self.assertFalse(det.feed(CHINESE_ONLY[i:i + 24]))

    def test_single_foreign_quote_does_not_trip(self) -> None:
        det = sc.SoupDetector()
        text = MIXED_QUOTE * 3
        for i in range(0, len(text), 24):
            det.feed(text[i:i + 24])
        self.assertFalse(det.tripped, det.stats())

    def test_replacement_chars_trip(self) -> None:
        det = sc.SoupDetector()
        blob = "texto " + ("\ufffd" * 40) + " mas texto normal" * 4
        for _ in range(4):
            det.feed(blob)
        self.assertTrue(det.tripped, det.stats())

    def test_two_scripts_with_latin_does_not_trip(self) -> None:
        det = sc.SoupDetector()
        text = ("The report saysdata صفحة 页面 content and more english words here "
                "with latin text between the arabic and han characters صفحة 页面 ") * 6
        for i in range(0, len(text), 24):
            det.feed(text[i:i + 24])
        self.assertFalse(det.tripped, det.stats())

    def test_han_and_punctuation_does_not_trip(self) -> None:
        det = sc.SoupDetector()
        text = ("Documento técnico 页面 † 页面 con texto latino alrededor y simbolos sueltos ") * 6
        for i in range(0, len(text), 24):
            det.feed(text[i:i + 24])
        self.assertFalse(det.tripped, det.stats())

    def test_no_text_no_trip(self) -> None:
        det = sc.SoupDetector()
        self.assertFalse(det.feed(""))
        self.assertFalse(det.tripped)

    def test_stats_shape(self) -> None:
        det = sc.SoupDetector()
        det.feed(SPANISH)
        stats = det.stats()
        for key in ("window_chars", "scripts", "script_count", "latin_ratio", "replacement_ratio"):
            self.assertIn(key, stats)


if __name__ == "__main__":
    unittest.main()
