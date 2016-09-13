GWAS - genome wide association study.

Format danych wejsciowych (liczby oddzielone spacja w kazdej lini)
pierwsza linia:
num_objects num_vars result_size a_priori
opisujaca kolejno liczbe obiektow, liczbe zmiennych

druga linia:
num_vars zmiennych decyzyjnych (0 lub 1)

kolejne num_vars lini (numerowanych od i=0 do num_vars) zawiera:
num_objects zmiennych opisowych (0, 1 lub 2) (numerowanych od j=0 do num_objects)
gdzie zmienna opisowa var_ij opisuje zmienna dla wartosci opisowej numer i oraz
obiektu numer j.



Aby odpalic testy nalezy uzyc komendy ./test.sh.
Mozliwe jest ze beda male roznice, spowodowane losowym wygenerowaniem 
mniejszej liczby wartosci (poprzez obliczenie thresholdu) lub male 
zamienienie kilku wartosci spowodowane bledem precyzji.
