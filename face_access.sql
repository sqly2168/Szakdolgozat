CREATE DATABASE IF NOT EXISTS face_access;
USE face_access;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(50) NOT NULL,
    employee_id VARCHAR(8) UNIQUE NOT NULL,
    id_card_number VARCHAR(20) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone_number VARCHAR(15),
    role ENUM('admin', 'employee', 'guest') NOT NULL DEFAULT 'employee',
    folder_link VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE worklogs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    exit_time DATETIME NULL,
    status ENUM('active', 'inactive') NOT NULL DEFAULT 'inactive',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

INSERT INTO users (full_name, employee_id, id_card_number, email, phone_number, role, folder_link) VALUES
('Kiss Péter', '12345', 'AA123456', 'peter.kiss@example.com', '+36201234567', 'employee', 
 'file:///C:/Users/SQLY/Desktop/egyi/6F/Szakdolgozat/data/employees/12345'),
('Nagy Anna', '67890', 'BB987654', 'anna.nagy@example.com', '+36207654321', 'admin', 
 'file:///C:/Users/SQLY/Desktop/egyi/6F/Szakdolgozat/data/employees/67890');

-- Ha van inaktív bejegyzés (az illető kilépett), akkor frissítjük és beléptetjük (active státusz)
UPDATE worklogs 
SET status = 'active', entry_time = NOW(), exit_time = NULL
WHERE user_id = (SELECT id FROM users WHERE employee_id = '12345') 
AND status = 'inactive'
ORDER BY id DESC
LIMIT 1;

-- Ha nincs még bejegyzése, akkor új aktív sor beszúrása
INSERT INTO worklogs (user_id, status) 
SELECT id, 'active' FROM users 
WHERE employee_id = '12345' 
AND NOT EXISTS (
    SELECT 1 FROM worklogs 
    WHERE user_id = (SELECT id FROM users WHERE employee_id = '12345') 
    AND status = 'active'
);

-- Kilépéskor egy új inaktív bejegyzés létrehozása az exit_time mentésével
INSERT INTO worklogs (user_id, exit_time, status)
SELECT id, NOW(), 'inactive' FROM users 
WHERE employee_id = '12345' 
AND EXISTS (
    SELECT 1 FROM worklogs 
    WHERE user_id = (SELECT id FROM users WHERE employee_id = '12345') 
    AND status = 'active'
);
