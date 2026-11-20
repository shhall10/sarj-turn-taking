"""Word layer: what the caller's words (from any streaming recogniser) say about their intent.

Saudi / Gulf / Levantine / Egyptian everyday Arabic is covered, written the way recognisers usually spell it
(with and without hamza, taa marbuta, tatweel and diacritics). `classify()` returns one of:

    STOP      the words show the caller wants the floor: "لحظة", "لا لا", "اسمع", "بس ...", "وقف", a question...
    KEEP      only acknowledgements / fillers so far: "إيه", "أها", "مم", "طيب", "صح", "زين", "هيه", "أيوه"
    HOLD_EOT  (agent silent) the caller is mid-thought: ends with a filler, a connector, or reads digits
    NONE      not enough evidence either way

Everything is pure functions so it can be unit-tested and reused by a recogniser running in another thread.
"""
from __future__ import annotations

import re

# --- normalisation ------------------------------------------------------------------------------------------------
_DIAC = re.compile(r"[ً-ْٰـ]")          # harakat, dagger alef, tatweel
_PUNCT = re.compile(r"[^\w\s؀-ۿ]|[،؛؟…\.\!\?,;:\"'«»]")


def normalize(text: str) -> str:
    t = _DIAC.sub("", text or "")
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = _PUNCT.sub(" ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)                     # هههههه -> هه, اممممم -> امم
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


# --- lexicons (normalised spelling) -------------------------------------------------------------------------------
# Acknowledgements: the caller is listening, not asking for the floor.
BACKCHANNEL = {
    "ايه", "اي", "ايوه", "ايوا", "ايوة", "هيه", "هي", "اها", "اه", "اهاه", "امم", "مم", "ام", "همم", "اوكي", "اوك",
    "طيب", "صح", "صحيح", "زين", "تمام", "ماشي", "اكيد", "نعم", "يب", "يس", "والله", "صادق", "معك", "معاك", "معايا",
    "فاهم", "فاهمه", "واضح", "حلو", "عادي", "اوه", "اها", "ايوه", "هلا", "ههه", "هه", "يا", "سلامات", "الله",
    "مضبوط", "بالضبط", "صار", "طبعا", "يعطيك", "العافيه", "تفضل", "تفضلي", "كمل", "كملي", "اسمعك",
}
# Fillers: thinking sounds. During the agent: not a floor claim. After the caller spoke: they are not done.
FILLER = {"امم", "مم", "ام", "اه", "اا", "يعني", "اممم", "همم", "اه", "شلون", "وش", "هذا", "ذا"}

# Words that on their own announce a floor claim ("hold on", "no no", "listen", "stop", "wait", "but").
STOP_WORDS = {
    "لحظه", "لحظة", "دقيقه", "ثانيه", "ثواني", "استنى", "استني", "انتظر", "انتظري", "وقف", "وقفي", "اوقف", "اسكت",
    "اسمع", "اسمعي", "اسمعني", "اسمعيني", "شوف", "شوفي", "خلاص", "بس", "لا", "لاء", "مو", "مب", "مش", "موب", "ماني",
    "ما", "غلط", "عفوا", "سوري", "لو", "ممكن", "ابغى", "ابي", "ابغي", "بغيت", "ودي", "عندي", "سؤال", "طيب؟",
    "كم", "متى", "وين", "ليش", "ليه", "كيف", "شلون", "وش", "ايش", "هل", "مين", "اصلا", "يا", "اخي", "اختي",
    "الو", "الغي", "الغاء", "غير", "اغير", "احجز", "احجزه", "احجزيه", "زفت",
}
# Phrases (normalised, space separated) that are a floor claim even when every word is ambiguous.
STOP_PHRASES = [
    "لا لا", "لحظه لحظه", "بس بس", "خلاص خلاص", "اسمع اسمع", "دقيقه بس", "ثانيه بس", "لحظه بس", "مو هذا",
    "مب هذا", "مش هذا", "ما طلبت", "ما ابي", "ما ابغى", "ما يناسب", "ما يناسبني", "مو مناسب", "ابغى اغير",
    "ابي اغير", "لو سمحت", "لو سمحتي", "عندي سؤال", "يا اخي", "يا اختي", "قصدي", "مو كذا", "مب كذا",
    "ما فهمت", "ما سمعت", "عيد", "عيدي", "كرر", "زين احجزه", "تمام احجزه", "اوكي احجزه",
]
# Strong single words: enough for an immediate STOP even as the very first word.
STRONG = {"لحظه", "استنى", "استني", "انتظر", "انتظري", "وقف", "وقفي", "اوقف", "اسمع", "اسمعي", "اسمعني",
          "اسمعيني", "دقيقه", "ثانيه", "اسكت", "الو"}
# Words that, when the caller's turn ends with them, mean "not finished yet".
CONTINUERS = {"و", "بس", "يعني", "لكن", "او", "اذا", "لان", "عشان", "علشان", "بعدين", "ثم", "اللي", "في", "على",
              "من", "مع", "الى", "حق", "تقريبا", "امم", "مم", "ام", "اه", "هو", "هي", "انا"}
DIGITS = {"صفر", "واحد", "اثنين", "ثنين", "ثلاثه", "ثلاث", "اربعه", "اربع", "خمسه", "خمس", "سته", "ست", "سبعه",
          "سبع", "ثمانيه", "ثمان", "تسعه", "تسع", "عشره", "عشر"}


def _bare(w: str) -> str:
    """Strip common clitics so 'والله' stays, but 'وبس' -> 'بس', 'فلحظة' -> 'لحظه'."""
    for p in ("و", "ف"):
        if len(w) > 3 and w.startswith(p) and w[1:] in STOP_WORDS | STRONG:
            return w[1:]
    return w


def classify(text: str, agent_talking: bool = True) -> str:
    """Intent from the words heard so far in the current caller burst."""
    raw = text or ""
    toks = [_bare(w) for w in tokens(raw)]
    if not toks:
        return "NONE"
    norm = " ".join(toks)
    if agent_talking:
        if any(w in STRONG for w in toks):
            return "STOP"
        if any(re.search(rf"(^| ){re.escape(p)}( |$)", norm) for p in STOP_PHRASES):
            return "STOP"
        if "؟" in raw or "?" in raw:
            return "STOP"
        content = [w for w in toks if w not in BACKCHANNEL and w not in FILLER]
        if not content:
            return "KEEP"
        stops = [w for w in content if w in STOP_WORDS]
        # one stop word plus anything else, or two content words: a real turn is starting
        if stops and (len(toks) >= 2 or stops[0] in {"لا", "بس", "خلاص"}):
            return "STOP"
        if len(content) >= 2:
            return "STOP"
        return "NONE"
    # agent silent: is the caller finished?
    last = toks[-1]
    if last in CONTINUERS or last in FILLER:
        return "HOLD_EOT"
    if sum(w in DIGITS for w in toks[-4:]) >= 2:
        return "HOLD_EOT"                         # reading out a number: pauses are not the end
    return "NONE"
