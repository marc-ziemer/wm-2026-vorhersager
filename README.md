# wm-2026-vorhersager
ML-Modell zur Vorhersage von WM-Spielausgängen
WM 2026 Vorhersager

Ein kleines Machine-Learning-Projekt, mit dem ich versuche, Ausgänge von Länderspielen vorherzusagen, mit Blick auf die WM 2026. Ich habe es diesen Sommer als Lernprojekt gebaut, um mir Grundlagen im Bereich Data Science und Machine Learning selbst beizubringen.

Worum es geht

Die Idee ist simpel: Kann man aus der bisherigen Form zweier Nationalmannschaften vorhersagen, wer ein Spiel gewinnt? Also nicht Unentschieden oder Zufall, sondern wirklich mit Daten arbeiten und schauen, ob ein Modell daraus etwas lernen kann.

Das Modell sagt für ein Spiel eine von drei Klassen voraus: Heimsieg, Auswärtssieg oder Unentschieden.

Daten

Ich nutze den Datensatz "International football results from 1872 to 2024" von Kaggle. Er enthält über 47.000 Länderspiele mit Datum, beiden Teams, Endergebnis und Turnier.

Link: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

Die Rohdaten liegen bewusst nicht in diesem Repo, weil sie recht groß sind und nicht mir gehören. Wer den Code selbst ausführen will, lädt sich results.csv von Kaggle runter.

Wie das Modell zu seinen Daten kommt

Das eigentlich interessante Problem war nicht das Modell selbst, sondern die Daten so aufzubereiten, dass sie überhaupt Sinn ergeben.

Zielvariable: Aus home_score und away_score wird für jedes Spiel bestimmt, ob es ein Heimsieg, Auswärtssieg oder Unentschieden war.

Form-Feature: Für jedes Team wird zu jedem Spielzeitpunkt die Durchschnittsform aus den letzten 20 Partien berechnet, und zwar nur aus Spielen, die vor dem jeweiligen Spiel stattfanden. Das war mir wichtig, weil man sonst Informationen aus der Zukunft ins Modell einschleust, was die Ergebnisse komplett verfälschen würde. Für ein Spiel aus dem Jahr 2010 zählen also nur Spiele, die auch wirklich vor 2010 stattfanden.

Form wird über ein einfaches Punktesystem berechnet: 3 Punkte für einen Sieg, 1 für Unentschieden, 0 für eine Niederlage, gemittelt über die letzten 20 Spiele. Ich habe mich bewusst für den Durchschnitt statt der Summe entschieden, damit Teams mit weniger Spielhistorie nicht automatisch schlechter dastehen.

Teams, für die noch keine 20 Vorgängerspiele existieren (also am Anfang des Datensatzes), bekommen keinen Formwert und werden aus dem Trainingsdatensatz entfernt.

Modell

Ich vergleiche zwei Modelle aus scikit-learn, einen Decision Tree und einen Random Forest. Als Baseline dient die einfachste denkbare Strategie: immer auf Heimsieg tippen. Das liegt bei etwa 49 Prozent, weil Heimteams im Fußball tatsächlich häufiger gewinnen.

Mit den beiden Form-Features (Heim-Form und Auswärts-Form) kommt das Modell auf rund 52,5 Prozent Genauigkeit. Das klingt erstmal nicht nach viel, ist aber ein echter Gewinn gegenüber der Baseline, gerade weil Fußballergebnisse notorisch schwer vorherzusagen sind.

Was nicht funktioniert hat

Ich habe versucht, das Modell mit zusätzlichen Features zu verbessern, unter anderem mit der durchschnittlichen Tordifferenz der letzten Spiele. Die Erwartung war, dass mehr Information zu einer besseren Vorhersage führt.

Das Gegenteil war der Fall. Mit den zusätzlichen Features sank die Genauigkeit auf etwa 50 Prozent. Vermutlich, weil die neuen Features stark mit den bestehenden korrelierten und dem Modell eher Rauschen als neue Erkenntnisse lieferten. Das war für mich der interessanteste Teil des Projekts: die Erkenntnis, dass mehr Features nicht automatisch ein besseres Modell bedeuten. Das schlankere Modell mit nur zwei Features bleibt deshalb die finale Version.

Dateien

WM Vorhersager.py enthält die komplette Pipeline von den Rohdaten bis zu den fertigen Features, inklusive Speichern der aufbereiteten Daten.

WM Vorhersager schnell.py lädt die bereits aufbereiteten Daten, trainiert das Modell und enthält eine Funktion, mit der sich beliebige Länderspiel-Paarungen vorhersagen lassen.

Grenzen des Projekts

Das Modell arbeitet nur mit zwei Features und ignoriert viele Faktoren, die Fußballspiele tatsächlich beeinflussen, etwa Verletzungen, Aufstellungen oder die Bedeutung des Spiels. Eine Genauigkeit von gut über 50 Prozent ist für so ein simples Modell in Ordnung, aber weit von einer verlässlichen Vorhersage entfernt. Das war auch nicht das Ziel, sondern eher ein Weg, mir die Grundlagen von Feature Engineering, sauberer Datenaufbereitung ohne Zukunftsdaten und dem Trainieren von Klassifikationsmodellen beizubringen.

Verwendete Werkzeuge

Python, pandas, scikit-learn
