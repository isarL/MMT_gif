numpy.random.randint(low, high=None, size=None, dtype='l')
deze code gaat een random getal of tuple van getallen genereren.
low = kleinste integer dat kan gekozen worden
high = hoogste integer dat kan gekozen worden
size = int of tuple of ints, bij bv (m, n , k) word een m*n*k tuple gemaakt van random waarden
dtype = type van het resultaat bv int int64 etc...

self.color_table_size = 2 ** BITS

Er wordt dus een color table aangemaakt van 2**BITS rijen en 3 kolommen. Per rij wordt er een random 3-tuple gegenereerd via numpy.random.randint met elk van de drie waarden tussen 0 en 255. Zo een tuple stelt op die manier een kleur voor. Met deze random gekozen kleuren uit de color table wordt de afbeelding ingekleurd, volgens het principe van van dichtsbijzinde kleur (minst verschil in waarden) van de oorspronkelijke kleur. Doordat de kleuren uit de color-table random gekozen zijn, zijn de kleuren in de bekomen afbeelding ook random, dus niet per se gelijk aan de oorspronkelijke. Hierdoor zal er dus zeer zelden (of zo goed als nooit) een ideaal resultaat weergegeven worden.

Hoe hoger het aantal bits hoe beter de waarden en hoe hoger de PSNR.
Dit is omdat een hoger aantal bits zal zorgen voor een groter kleurenpalet wat sowieso voor betere kleurovergangen zal zorgen.
Bij het random kiezen van kleuren kan je pech hebben dat er een bepaalde kleur wordt gekozen die maar erg weinig voorkomt,
bij een groter kleurenpalet zal dit een kleinere impact hebben omdat veelvoorkomende kleuren natuurlijk veel meer gekozen zullen worden, en de kans dat alle kleuren slecht gekozen zijn dus zeer klein wordt.
Bij 8 bits is het verschil met de orginele foto al zeer klein.
