Het zwart in de afbeelding is gewijzigd naar appelblauwzeegroen, voor de rest zijn de afbeeldingen gelijk. De verklaring hiervoor is dat de color-table maar 2 rijen bevat en dus dat er enkel voor de rode waarden een median-cut wordt uitgevoerd. De waarden in de color-table worden dan [0,255,255] en [255,255,255], waardoor het zwart gewijzigd wordt. Ook zijn de bestandsgroottes groter voor median-cut, aangezien zwart, waarde [0,0,0] gewijzigd is in [0,255,255] en deze pixels dus een grotere waarde hebben en dus meet geheugen nodig hebben om opgeslagen te worden.

PSNR???
