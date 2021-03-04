1)	Steek eerst de rgb waarden in 3 gesorteerde lijst.
	roep nu divide op .
	als d (divide) = 1
		return de kleur die het best dit blok weergeeft
	anders
		bereken van elke kleur de range, en zoek de kleur met de grootste range.
		verdeel van het kleur met de grootste range enkel deze gesorteerde lijst in 2,
		bereken nu recursief divide 2x opnieuw, elk met de helft van de gesplitste lijst + de andere 2 lijsten en d//2

2) Het algoritme gaat gewoon door, het kan dus dat er meerdere malen hetzelfde kleur voorkomt in het kleurenpalet.

3) Het kleur dat als r, g en b waarde de mediaan heeft van elk van deze lijsten.

4) Geen enkel aangezien je steeds splitst op de mediaan en die erna op de rand ligt van de nieuwe blokken.

