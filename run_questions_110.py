import os
from pathlib import Path

# Force demo setup to run inside project root
BASE = Path(__file__).parent
os.environ.setdefault("RAG_OFFLINE", "1")

from rag_final import setup_rag_system_for_demo


def load_questions(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


EXPECTED = {
    "czy kłębonogi trójoczne zamieszkują planetę nimbara-12?": "Tak",
    "czy jerund phales zaproponował nazwę gatunkową kłębonogów w 2803 roku?": "Tak",
    "czy ekspedycja kapitana osmoula miała miejsce w 2834 roku?": "Tak",
    "czy doktor mirinda quay odkryła błąd phalesa w 2856 roku?": "Tak",
    "czy piewcy srebrnego zmierzchu zostali założeni przez orthala mendrakusa?": "Tak",
}

CONTAINS = {
    "czy kłębonogi trójoczne są ssakami, grzybami symbiotycznymi, czy może obiema tymi formami życia jednocześnie, i od jakich czynników zależy ich klasyfikacja taksonomiczna?": [
        "ssak", "grzyb", "temperatur"
    ],
    "w jaki sposób można zweryfikować autentyczność wizji orthala mendrakusa dotyczących \"bramek nierodzenia\" i jego telekomunikacji z vorthakulem nieskończonym, skoro bazowały one na transach wywołanych przez fermentowane glony gatunku algae transcendentia?": [
        "algae", "trans", "weryfik"
    ],
    "jeśli jerund phales zaproponował nazwę gatunkową w 2803 roku, pomyłka została odkryta w 2856 roku, a kolegium etnoeksobiolologów zakwestionowało badania quay w nieokreślonym czasie, ale vold-quex rozpoczęła swoje 23-letnie badania prowadzące do publikacji raportu w 2877 roku – to kiedy dokładnie rozpoczęła swoje badania i czy mogła uwzględnić odkrycie quay z 2856 roku?": [
        "2854", "2856"
    ],
    "czy istnieje fundamentalna różnica epistemologiczna między badaniem istot, które mogą odżywiać się \"abstrakcyjnymi koncepcjami takimi jak zapomniane wspomnienia lub niezrealizowane możliwości\" a badaniem istot żywiących się zwykłą materią organiczną, i czy nasze metody naukowe są w ogóle odpowiednie do badania pierwszego typu fenomenów?": [
        "substrat", "ontologic", "metod", "organicz"
    ],
}


def evaluate(question: str, answer: str):
    ql = question.lower().strip()
    response = (answer or "").strip()
    if ql in EXPECTED:
        expected_prefix = EXPECTED[ql]
        return response.lower().startswith(expected_prefix.lower()), f"expect {expected_prefix}"
    required = CONTAINS.get(ql)
    if required:
        lower = response.lower()
        missing = [fragment for fragment in required if fragment not in lower]
        return len(missing) == 0, ("missing: " + ", ".join(missing) if missing else "ok")
    return True, "no strict expectation"


def main():
    rag = setup_rag_system_for_demo(enable_ab_testing=False)
    questions_path = BASE / "questions" / "pytania110.txt"
    prompts = load_questions(questions_path)

    total = 0
    passed = 0
    for idx, prompt in enumerate(prompts, 1):
        result = rag.query(prompt)
        answer = result["answer"]
        ok, note = evaluate(prompt, answer)
        total += 1
        if ok:
            passed += 1
        print(f"{idx:02d}. {'PASS' if ok else 'FAIL'} | {prompt}\n   -> {answer}\n   [{note}]\n")

    print(f"SUMMARY: {passed}/{total} passed")


if __name__ == "__main__":
    main()
