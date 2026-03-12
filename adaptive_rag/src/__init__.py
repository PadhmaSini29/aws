import os
import sqlite3

DB_PATH = r".\data\wealthmanagement.db"

os.makedirs(".\\data", exist_ok=True)

schema_sql = """
DROP TABLE IF EXISTS portfolio_performance;
DROP TABLE IF EXISTS investment;
DROP TABLE IF EXISTS client;

CREATE TABLE client (
    client_id INTEGER PRIMARY KEY NOT NULL,
    first_name TEXT,
    last_name TEXT,
    age INTEGER,
    risk_tolerance TEXT
);

CREATE TABLE investment (
    investment_id INTEGER PRIMARY KEY NOT NULL,
    client_id INTEGER,
    asset_type TEXT,
    investment_amount REAL,
    current_value REAL,
    purchase_date VARCHAR,
    FOREIGN KEY (client_id) REFERENCES client(client_id)
);

CREATE TABLE portfolio_performance (
    client_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    total_return_percentage REAL,
    benchmark_return REAL,
    PRIMARY KEY (client_id, year),
    FOREIGN KEY (client_id) REFERENCES client(client_id)
);

INSERT INTO client (client_id, first_name, last_name, age, risk_tolerance) VALUES
(1, 'John', 'Smith', 45, 'Conservative'),
(2, 'Sarah', 'Johnson', 38, 'Moderate'),
(3, 'Michael', 'Brown', 52, 'Aggressive'),
(4, 'Emily', 'Davis', 29, 'Moderate'),
(5, 'Robert', 'Wilson', 61, 'Conservative');

INSERT INTO investment (investment_id, client_id, asset_type, investment_amount, current_value, purchase_date) VALUES
(1, 1, 'Bonds', 50000.00, 52000.00, '2023-01-15'),
(2, 1, 'Stocks', 30000.00, 35000.00, '2023-03-10'),
(3, 2, 'Stocks', 75000.00, 82000.00, '2022-06-20'),
(4, 2, 'ETF', 25000.00, 27500.00, '2023-02-05'),
(5, 3, 'Stocks', 100000.00, 125000.00, '2022-01-10'),
(6, 3, 'Options', 20000.00, 18000.00, '2023-05-15'),
(7, 4, 'Mutual Funds', 40000.00, 43000.00, '2023-01-20'),
(8, 5, 'Bonds', 80000.00, 81000.00, '2022-12-01');

INSERT INTO portfolio_performance (client_id, year, total_return_percentage, benchmark_return) VALUES
(1, 2022, 8.5, 7.2),
(1, 2023, 12.3, 10.1),
(2, 2022, 15.2, 12.8),
(2, 2023, 18.7, 15.3),
(3, 2022, 22.1, 18.5),
(3, 2023, 25.8, 20.2),
(4, 2022, 11.4, 9.8),
(4, 2023, 14.6, 12.1),
(5, 2022, 6.8, 5.9),
(5, 2023, 7.2, 6.5);
"""

conn = sqlite3.connect(DB_PATH)
try:
    conn.executescript(schema_sql)
    conn.commit()
    print(f"✅ Created DB at {DB_PATH}")
finally:
    conn.close()
