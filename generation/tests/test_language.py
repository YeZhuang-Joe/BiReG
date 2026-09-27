import unittest
from bireg.language import detect

class RoutingTests(unittest.TestCase):
    def test_chinese(self):self.assertEqual(detect('左边一个红色杯子，右边一个蓝色碗。')['selected_language'],'zh')
    def test_english(self):self.assertEqual(detect('A red cup to the left of a blue bowl.')['selected_language'],'en')
    def test_english_with_chinese_sign(self):self.assertEqual(detect('A shop with a sign reading “茶馆”.')['selected_language'],'en')
    def test_chinese_with_english_sign(self):self.assertEqual(detect('一家咖啡馆，招牌写着 “OPEN”。')['selected_language'],'zh')
    def test_brand_mixed(self):self.assertEqual(detect('一位穿着 Nike 外套的女孩，背景为 cyberpunk 风格')['selected_language'],'zh')
    def test_ambiguous(self):self.assertTrue(detect('红色 red blue green')['requires_manual_language'])
    def test_manual_override(self):
        r=detect('红色 red blue green','zh');self.assertEqual(r['selected_language'],'zh');self.assertIsNone(r['automatic_suggestion'])
    def test_symbols(self):self.assertIsNone(detect('12345 !?!')['selected_language'])
    def test_quote_fallback(self):
        r=detect('“茶馆”');self.assertEqual(r['selected_language'],'zh');self.assertIn('fallback',r['count_basis'])
    def test_apostrophe(self):self.assertEqual(detect("A boy's coat isn't red.")['selected_language'],'en')
    def test_urls(self):self.assertIsNone(detect('https://example.com/123')['selected_language'])
    def test_original_unchanged(self):
        p='Ａ red cup，招牌“茶馆”';old=p;detect(p);self.assertEqual(p,old)
    def test_threshold_exact(self):
        self.assertEqual(detect('甲乙丙丁戊己庚 one two three')['selected_language'],'zh')
        self.assertEqual(detect('甲乙丙 one two three four five six seven')['selected_language'],'en')

if __name__=='__main__':unittest.main()
