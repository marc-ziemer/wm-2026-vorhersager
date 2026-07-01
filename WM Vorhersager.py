
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- Daten laden ---
ergebnisse = pd.read_csv(
    r'C:\Users\irina\OneDrive\Bilder\Desktop\Praktikum erreichen\results.csv')

# --- Daten bereinigen + Datum umwandeln ---
ergebnisse = ergebnisse.dropna(subset=['home_score', 'away_score'])
ergebnisse['date'] = pd.to_datetime(ergebnisse['date'])


# --- Zielvariable: wer hat gewonnen? ---
def Winners(zeile):
    if zeile['home_score'] > zeile['away_score']:
        return 'Sieg Heimmannschaft'
    elif zeile['home_score'] < zeile['away_score']:
        return 'Sieg Auswärtsmannschaft'
    else:
        return 'Unentschieden'


ergebnisse['result'] = ergebnisse.apply(Winners, axis=1)


# --- Baustein: Punkte für ein Team in einem Spiel ---
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


def Tordifferenz(zeile, team):
    if team == zeile['home_team']:
        return zeile['home_score'] - zeile['away_score']
    else:
        return zeile['away_score'] - zeile['home_score']
# --- Form: Durchschnittspunkte der letzten 20 Spiele VOR einem Datum ---


def tor_form(team, datum):
    datum = pd.to_datetime(datum)
    matches = ergebnisse[
        ((ergebnisse['home_team'] == team) | (ergebnisse['away_team'] == team))
        & (ergebnisse['date'] < datum)
    ]
    last_20 = matches.sort_values(by="date", ascending=False).head(20)
    tordifferenz = last_20.apply(
        lambda zeile: Tordifferenz(zeile, team), axis=1)
    return tordifferenz.mean()


def form(team, datum):
    datum = pd.to_datetime(datum)
    matches = ergebnisse[
        ((ergebnisse['home_team'] == team) | (ergebnisse['away_team'] == team))
        & (ergebnisse['date'] < datum)
    ]
    last_20 = matches.sort_values(by="date", ascending=False).head(20)
    punkte = last_20.apply(lambda zeile: Punkte(zeile, team), axis=1)
    return punkte.mean()


ergebnisse["heim_form"] = ergebnisse.apply(
    lambda zeile: form(zeile["home_team"], zeile["date"]), axis=1)
ergebnisse["auswärts_form"] = ergebnisse.apply(
    lambda zeile: form(zeile["away_team"], zeile["date"]), axis=1)

ergebnisse["heim_tordiff"] = ergebnisse.apply(
    lambda zeile: tor_form(zeile["home_team"], zeile["date"]), axis=1)
ergebnisse["auswärts_tordiff"] = ergebnisse.apply(
    lambda zeile: tor_form(zeile["away_team"], zeile["date"]), axis=1)

print(ergebnisse[['date', 'home_team', 'away_team',
      'heim_form', 'auswärts_form', 'result']].head(30))
ergebnisse = ergebnisse.dropna(
    subset=['heim_form', 'auswärts_form', 'heim_tordiff', 'auswärts_tordiff'])
ergebnisse['form_differenz'] = ergebnisse['heim_form'] - \
    ergebnisse['auswärts_form']
ergebnisse.to_csv(
    r'C:\Users\irina\OneDrive\Bilder\Desktop\Praktikum erreichen\ergebnisse_mit_form.csv', index=False)
print(ergebnisse.shape)
X = ergebnisse[["heim_form", "auswärts_form"]]
y = ergebnisse["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(ergebnisse['result'].value_counts(normalize=True))
modell = RandomForestClassifier(random_state=42)
modell.fit(X_train, y_train)
print(modell.score(X_test, y_test))
