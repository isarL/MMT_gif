Aangezien de afbeeldingen row-major in het geheugen zitten, wordt er eerst geïtereerd over de rijen en pas daarna over de kolommen. Zo zijn de geheugentoegangen sneller, dan als er eerst over de kolommen en dan pas over de rijen zou geïtereerd worden. Ook wordt er gebruik gemaakt van geneste if-constructies, aangezien bij sommige sowieso de buitenste if moet gelden om de binnenste mogelijk te maken. Zo worden sommige vergelijkingen vermeden voor bepaalde elementen.