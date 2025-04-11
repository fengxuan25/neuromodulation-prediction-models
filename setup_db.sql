-- Initial setup database
-- # Log in as root (you'll be prompted for your password)
# mysql -u root -p
# CREATE DATABASE neuroscience_experiments;
# CREATE USER 'neuro_user'@'localhost' IDENTIFIED BY 'your_password';
# GRANT ALL PRIVILEGES ON neuroscience_experiments.* TO 'neuro_user'@'localhost';
# FLUSH PRIVILEGES;
# EXIT;

-- Create database and user
CREATE DATABASE IF NOT EXISTS neuroscience_experiments;
USE neuroscience_experiments;

-- Create animals table
CREATE TABLE IF NOT EXISTS animals (
    animal_id VARCHAR(20) PRIMARY KEY,
    strain VARCHAR(50),
    sex CHAR(1),
    birth_date DATE,
    additional_info TEXT
);

-- Create sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id INT AUTO_INCREMENT PRIMARY KEY,
    animal_id VARCHAR(20),
    session_date DATETIME,
    session_number INT,
    experiment_type VARCHAR(50),
    FOREIGN KEY (animal_id) REFERENCES animals(animal_id)
);

-- Create behavioral_data table
CREATE TABLE IF NOT EXISTS behavioral_data (
    data_id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT,
    time_point FLOAT,
    position FLOAT,
    velocity FLOAT,
    licking TINYINT,
    reward TINYINT,
    grab_ach_signal FLOAT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Create cells table
CREATE TABLE IF NOT EXISTS cells (
    cell_id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT,
    cell_number INT,
    cell_type VARCHAR(50),
    region VARCHAR(50),
    is_place_cell TINYINT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Create calcium_activity table
CREATE TABLE IF NOT EXISTS calcium_activity (
    activity_id INT AUTO_INCREMENT PRIMARY KEY,
    cell_id INT,
    time_point FLOAT,
    activity_value FLOAT,
    FOREIGN KEY (cell_id) REFERENCES cells(cell_id)
);

-- Create indexes for better performance
CREATE INDEX idx_behavioral_session_time ON behavioral_data(session_id, time_point);
CREATE INDEX idx_calcium_cell_time ON calcium_activity(cell_id, time_point);
CREATE INDEX idx_cells_session ON cells(session_id);