import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- Fertige Daten MIT Form laden (schnell, keine Neuberechnung!) ---
ergebnisse = pd.read_csv(
    r'C:\Users\irina\OneDrive\Bilder\Desktop\Praktikum erreichen\ergebnisse_mit_form.csv')

# date wieder zu echtem Datum machen (geht beim Speichern verloren)
ergebnisse['date'] = pd.to_datetime(ergebnisse['date'])


# --- Bausteine, die die Vorhersage-Funktion braucht ---
def Winners(zeile):
    if zeile['home_score'] > zeile['away_score']:
        return 'Sieg Heimmannschaft'
    elif zeile['home_score'] < zeile['away_score']:
        return 'Sieg Auswärtsmannschaft'
    else:
        return 'Unentschieden'


def Punkte(zeile, team):
    if team == zeile['home_team'] and zeile['result'] == 'Sieg Heimmannschaft':
        return 3
    elif team == zeile['away_team'] and zeile['result'] == 'Sieg Auswärtsmannschaft':
        return 3
    elif zeile['result'] == 'Unentschieden':
        return 1
    elif team == zeile['home_team'] and zeile['result'] == 'Sieg Auswärtsmannschaft':
        return 0
    elif team == zeile['away_team'] and zeile['result'] == 'Sieg Heimmannschaft':
        return 0


def form(team, datum):
    datum = pd.to_datetime(datum)
    matches = ergebnisse[
        ((ergebnisse['home_team'] == team) | (ergebnisse['away_team'] == team))
        & (ergebnisse['date'] < datum)
    ]
    last_20 = matches.sort_values(by="date", ascending=False).head(20)
    punkte = last_20.apply(lambda zeile: Punkte(zeile, team), axis=1)
    return punkte.mean()


# --- Features (X) und Ziel (y) ---
X = ergebnisse[["heim_form", "auswärts_form"]]
y = ergebnisse["result"]

# --- Train/Test-Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- Baseline zum Vergleich ---
print("Baseline (Verteilung der Ergebnisse):")
print(ergebnisse['result'].value_counts(normalize=True))

# --- Modell trainieren und bewerten ---
modell = RandomForestClassifier(random_state=42)
modell.fit(X_train, y_train)
print("\nModell-Genauigkeit:", modell.score(X_test, y_test))


# --- Vorhersage-Funktion ---
def vorhersage(heim_team, auswaerts_team):
    heute = "2026-07-01"
    h_form = form(heim_team, heute)
    a_form = form(auswaerts_team, heute)

    neue_daten = pd.DataFrame({
        "heim_form": [h_form],
        "auswärts_form": [a_form]
    })

    tipp = modell.predict(neue_daten)
    print(f"\n{heim_team} vs. {auswaerts_team}")
    print(f"Heim-Form: {h_form:.2f} | Auswärts-Form: {a_form:.2f}")
    print("Tipp:", tipp[0])


vorhersage("Germany", "France")
vorhersage("Brazil", "Argentina")
vorhersage("Spain", "Italy")