"""Deterministic Chinese/English template routing; never rewrites input text."""
import re
import unicodedata

VERSION='zh-en-script-routing-v1'
URL=re.compile(r'(?:https?://|www\.)[^\s<>"“”‘’，。！？；]+',re.I)
QUOTES=[re.compile(r'"[^"\n]*"'),re.compile(r'“[^”]*”'),re.compile(r'‘[^’]*’'),re.compile(r'「[^」]*」'),re.compile(r'『[^』]*』'),re.compile(r"(?<!\w)'[^'\n]*'(?!\w)")]
ENGLISH_WORD=re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")

def counts(text):
    han=sum(unicodedata.name(c,'').startswith(('CJK UNIFIED IDEOGRAPH','CJK COMPATIBILITY IDEOGRAPH')) for c in text)
    return han,len(ENGLISH_WORD.findall(text))

def detect(prompt,requested='auto'):
    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Prompt must be nonempty text')
    if requested not in ('auto','zh','en'):raise ValueError('language must be auto, zh or en')
    normalized=unicodedata.normalize('NFKC',prompt)
    without_urls=URL.sub(' ',normalized)
    reduced=without_urls
    for pattern in QUOTES:reduced=pattern.sub(' ',reduced)
    h,e=counts(reduced);basis='unquoted_text_without_urls'
    if h+e==0:
        h,e=counts(without_urls);basis='quoted_content_fallback_without_urls'
    total=h+e
    if not total:suggestion=None;reason='no_supported_script_evidence'
    elif 10*h>=7*total:suggestion='zh';reason='han_share_at_least_0.7'
    elif 10*h<=3*total:suggestion='en';reason='han_share_at_most_0.3'
    else:suggestion=None;reason='mixed_ambiguous'
    selected=suggestion if requested=='auto' else requested
    return dict(detector_version=VERSION,unicode_database_version=unicodedata.unidata_version,
        requested_language=requested,automatic_suggestion=suggestion,selected_language=selected,
        selection_source='manual_override' if requested!='auto' else ('automatic' if selected else 'manual_required'),
        reason=reason,han_characters=h,english_words=e,han_share=h/total if total else None,
        count_basis=basis,thresholds={'en_max':0.3,'zh_min':0.7},
        requires_manual_language=selected is None,original_prompt_modified=False,
        score_is_probability=False)
