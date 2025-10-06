import os
import sys
from pathlib import Path

# Ensure we run in project root of Rag2
BASE = Path(__file__).parent
os.environ.setdefault("RAG_OFFLINE", "1")  # force offline to avoid external API calls

from rag_final import setup_rag_system_for_demo


def load_questions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


EXPECTED = {
    # Yes/No
    "czy gzubry srebrnopióre potrafią latać?": "Nie",
    "czy gzubry zamieszkują planetę xelora-7?": "Tak",
    "czy dorosły gzubr może ważyć 180 kilogramów?": "Tak",
    "czy gzubry porozumiewają się wyłącznie za pomocą skrzeku?": "Nie",
    "czy pierwsze wzmianki o gzubrach pochodzą z roku 2772?": "Tak",
}

# For open-ended, we validate via required substrings
CONTAINS = {
    "jakie cechy charakterystyczne wyróżniają gzubry srebrnopióre?": [
        "bioluminesc", "zmienia", "wilgotno", "emoc", "nielotn", "xelora-7",
    ],
    "w jaki sposób gzubry komunikują się między sobą?": [
        "skrzek", "ultradźwię",
    ],
    "co to znaczy, że gzubry są istotami \"półinteligentnymi\"?": [
        "45", "zunar",
    ],
    "na czym polega niezwykły cykl życia gzubrów?": [
        "fotosynte", "zakorzen", "3 tyg",
    ],
    "jakie znaczenie kulturowe miały gzubry dla lokalnych plemion xeloran?": [
        "strażnic", "piór", "rytua", "fale mózg",
    ],
    "jaką maksymalną wysokość może osiągnąć dorosły gzubr?": [
        "2,4",  # accept comma variant; answer uses comma
    ],
    "od czego zależy zmiana koloru upierzenia gzubrów?": [
        "wilgotno", "emoc",
    ],
    "ile słów w języku zunarijskim potrafią rozpoznać gzubry?": [
        "45",
    ],
    "jak długo trwa faza fotosyntezy u gzubrów?": [
        "3 tyg",
    ],
    "w jakim roku korpus ekobiologii galaktycznej po raz pierwszy wspomniał o gzubrach?": [
        "2772",
    ],
    "dlaczego gzubry nazwano \"ptakoroślinnami\"?": [
        "nielotn", "fotosynte",
    ],
    "jakie mogą być przyczyny bioluminescencji upierzenia gzubrów?": [
        "wilgotno", "emoc",
    ],
    "jak nazwa \"strażnicy snów\" może łączyć się z właściwościami piór gzubrów?": [
        "fale mózg", "rytua",
    ],
    "jakie korzyści ewolucyjne może dawać gzubrom faza fotosyntezy?": [
        "energ", "przetrwan",
    ],
    "czym różni się półinteligencja gzubrów od pełnej inteligencji?": [
        "45", "ogranicz",
    ],
}


def evaluate(question: str, answer: str):
    ql = question.lower().strip()
    ans = answer.strip()
    if ql in EXPECTED:
        ok_prefix = EXPECTED[ql]
        return ans.lower().startswith(ok_prefix.lower()), f"expect {ok_prefix}"
    # contains check
    req = CONTAINS.get(ql)
    if req:
        low = ans.lower()
        missing = [k for k in req if k not in low]
        return (len(missing) == 0), ("missing: " + ", ".join(missing) if missing else "ok")
    # default: cannot auto-check
    return True, "no strict expectation"


def main():
    rag = setup_rag_system_for_demo(enable_ab_testing=False)

    qpath = BASE / "questions" / "pytania.txt"
    questions = load_questions(qpath)

    passed = 0
    total = 0
    for i, q in enumerate(questions, 1):
        res = rag.query(q)
        ans = res["answer"]
        ok, note = evaluate(q, ans)
        total += 1
        if ok:
            passed += 1
        print(f"{i:02d}. {'PASS' if ok else 'FAIL'} | {q}\n   -> {ans}\n   [{note}]\n")

    print(f"SUMMARY: {passed}/{total} passed")


if __name__ == "__main__":
    main()
