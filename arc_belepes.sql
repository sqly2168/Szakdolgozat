CREATE DATABASE arc_belepes;
USE arc_belepes;

CREATE TABLE felhasznalok (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nev VARCHAR(50) NOT NULL,
    torzsszam VARCHAR(8) UNIQUE NOT NULL
);

CREATE TABLE belepesek (
    id INT AUTO_INCREMENT PRIMARY KEY,
    felhasznalo_id INT NOT NULL,
    belepes_idopont DATETIME DEFAULT CURRENT_TIMESTAMP,
    kilepes_idopont DATETIME NULL,
    FOREIGN KEY (felhasznalo_id) REFERENCES felhasznalok(id) ON DELETE CASCADE
);
INSERT INTO felhasznalok (nev, torzsszam) VALUES
('Kiss Péter', '12345'),
('Nagy Anna', '67890');

INSERT INTO belepesek (felhasznalo_id) 
VALUES ((SELECT id FROM felhasznalok WHERE torzsszam = '12345'));

UPDATE belepesek 
SET kilepes_idopont = NOW() 
WHERE felhasznalo_id = (SELECT id FROM felhasznalok WHERE torzsszam = '12345') 
AND kilepes_idopont IS NULL;

select * from belepesek;
select * from felhasznalok;


