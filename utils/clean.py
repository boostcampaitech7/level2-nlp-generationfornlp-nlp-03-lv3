import re


class WikipediaTextPreprocessor:
    def __init__(self):
        pass

    def reduce_punctuation(self, text):
        return re.sub(r"\.{2,}", ".", text)

    def add_space_after_period(self, text):
        return re.sub(r"\.(\S)", r". \1", text)

    def remove_dates(self, text):
        text = re.sub(r"날짜=\d{4}-\d{2}-\d{2}(\|)?", "", text)  # '날짜=YYYY-MM-DD' 형식만 제거
        text = re.sub(r"date=[a-zA-Z가-힣\s]*\d+", "", text)  # 'date=' 형식만 제거
        return text

    def remove_text_equals(self, text):
        return re.sub(r"text=", "", text)

    def remove_citations(self, text):
        return re.sub(r"p=\d+([–-]\d+)?|pp=\d+([–-]\d+)?|p=not cited|pp=not cited|page=\d+|pages=\d+[–-]\d+", "", text)

    def remove_group_and_name(self, text):
        return re.sub(r"group=\w+\||name=\w+\|", "", text)

    def remove_link_yes(self, text):
        return re.sub(r"\|link=yes", "", text)

    def remove_thumbnail(self, text):
        return re.sub(r"섬네일\|.*?\|.*?\|.*?(\n|$)", "", text, flags=re.DOTALL)

    def remove_date_with_text(self, text):
        return re.sub(r"\|.*?\d{4}년 \d{1,2}월 \d{1,2}일", "", text)

    def remove_date_with_text_and_suffix(self, text):
        return re.sub(r"\|.*?\d{4}년 \d{1,2}월 \d{1,2}일자", "", text)

    def remove_broken_html_refs(self, text):
        text = re.sub(r"<ref.*?>.*?(</ref>|</REF>|(\n|$))", "", text, flags=re.DOTALL)
        return re.sub(r'ref name="cc1"\/>', "", text)

    def remove_orphan_closing_refs(self, text):
        return re.sub(r"([.?!])[^.?!]*?</ref>", r"\1", text)

    def remove_order_and_st(self, text):
        text = re.sub(r"order=t", "", text)
        return re.sub(r"\(s=.*?\|t=.*?\)", "", text)

    def remove_pipe_number_pipe(self, text):
        return re.sub(r"\|\d+\|", "", text)

    def remove_title_until_newline(self, text):
        return re.sub(r"제목=.*?(\n|$)", "", text)

    def remove_quote_until_newline(self, text):
        return re.sub(r"인용구=.*?(\n|$)", "", text)

    def remove_description(self, text):
        return re.sub(r"\|설명=.*?:", "", text)

    def replace_newline_with_space(self, text):
        return text.replace("\\n", " ").replace("\n", " ")

    def reduce_multiple_spaces(self, text):
        return re.sub(r"\s{2,}", " ", text)

    def preprocess_pipeline(self, text):
        text = self.reduce_punctuation(text)
        # text = self.add_space_after_period(text)
        text = self.remove_dates(text)
        text = self.remove_text_equals(text)
        text = self.remove_citations(text)
        text = self.remove_group_and_name(text)
        text = self.remove_link_yes(text)
        text = self.remove_thumbnail(text)
        # text = self.remove_date_with_text(text)
        # text = self.remove_date_with_text_and_suffix(text)
        text = self.remove_broken_html_refs(text)
        text = self.remove_orphan_closing_refs(text)
        text = self.remove_order_and_st(text)
        text = self.remove_pipe_number_pipe(text)
        text = self.remove_title_until_newline(text)
        text = self.remove_quote_until_newline(text)
        text = self.remove_description(text)
        text = self.replace_newline_with_space(text)
        text = self.reduce_multiple_spaces(text)

        return text
