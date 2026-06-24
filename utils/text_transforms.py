from transformers import AutoTokenizer
from pyvi import ViTokenizer    

class TextTransform:
    """
    Tokenize câu mô tả thành input cho BERT.

    BERT cần 2 tensor:
        - input_ids:      [max_len] — chỉ số token trong vocab BERT
        - attention_mask:  [max_len] — 1 = token thật, 0 = padding
    """

    def __init__(self, bert_model="", max_query_len=15):
        """
        Args:
            bert_model (str): Tên model BERT trên HuggingFace
            max_query_len (int): Số token text tối đa (KHÔNG tính [CLS], [SEP])
                Tổng chiều dài = max_query_len + 2 (cho [CLS] và [SEP])
        """
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        # max_query_len = 15 token text
        # + 2 special tokens ([CLS] + [SEP]) = 17 tokens tổng
        self.max_len = max_query_len + 2

    def __call__(self, text):
        """
        Tokenize 1 câu text.

        """

        encoded = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",      # Pad đến max_length
            truncation=True,           # Cắt nếu dài quá
            return_attention_mask=True,
            return_token_type_ids=False,  # Không cần (chỉ có 1 câu)
        )

        input_ids = encoded["input_ids"]           # list[int], len = max_len
        attention_mask = encoded["attention_mask"]  # list[int], len = max_len

        return input_ids, attention_mask

    def decode(self, input_ids):
        """
        Chuyển input_ids ngược lại thành text
        """
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)


# TEST
if __name__ == "__main__":
    print("=== Test TextTransform ===\n")

    transform = TextTransform(
        bert_model="bert-base-uncased",
        max_query_len=15
    )

    # Test 1: Câu bình thường
    text = "the man in red shirt"
    input_ids, attention_mask = transform(text)
    print(f"Input text:     '{text}'")
    print(f"Input IDs:      {input_ids}")
    print(f"Attention mask: {attention_mask}")
    print(f"Length:          {len(input_ids)} (= 15 + 2)")
    print(f"Decoded:        '{transform.decode(input_ids)}'")

    # Test 2: Câu dài (sẽ bị cắt)
    long_text = "the person wearing a blue hat standing near the large wooden table on the left side of room"
    input_ids_long, mask_long = transform(long_text)
    print(f"\nLong text:      '{long_text}'")
    print(f"Decoded (cut):  '{transform.decode(input_ids_long)}'")
    print(f"Mask sum:       {sum(mask_long)} tokens thật (max {len(mask_long)})")

    # Test 3: Câu ngắn (sẽ bị pad)
    short_text = "left dog"
    input_ids_short, mask_short = transform(short_text)
    print(f"\nShort text:     '{short_text}'")
    print(f"Input IDs:      {input_ids_short}")
    print(f"Attention mask: {mask_short}")
    print(f"Mask sum:       {sum(mask_short)} tokens thật (rest is PAD)")

    print("\n✅ TextTransform test passed!")
