import psycopg2

# 🔐 RDS credentials
DB_HOST = "vs-pg-db.crgee0iss08p.ap-south-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "postgres123"

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    print("✅ Connected to PostgreSQL")

except Exception as e:
    print("❌ Connection failed")
    print(e)
    exit()

cursor.execute("""
CREATE TABLE IF NOT EXISTS employees (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS skills (
    id SERIAL PRIMARY KEY,
    skill_name TEXT UNIQUE NOT NULL
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS employee_skills (
    employee_id INT REFERENCES employees(id) ON DELETE CASCADE,
    skill_id INT REFERENCES skills(id) ON DELETE CASCADE,
    years_of_experience INT NOT NULL,
    PRIMARY KEY (employee_id, skill_id)
);
""")

conn.commit()
print("✅ Tables created")


cursor.execute("""
INSERT INTO employees (name, email)
VALUES (%s, %s)
ON CONFLICT (email) DO NOTHING;
""", ("henry", "henry@example.com"))

cursor.execute("""
INSERT INTO employees (name, email)
VALUES (%s, %s)
ON CONFLICT (email) DO NOTHING;
""", ("tom", "tom@example.com"))

# Skills
skills = ["Python", "SQL", "AWS"]
for skill in skills:
    cursor.execute("""
    INSERT INTO skills (skill_name)
    VALUES (%s)
    ON CONFLICT (skill_name) DO NOTHING;
    """, (skill,))

conn.commit()
print("✅ Employees & skills inserted")

# ================================
# MAP SKILLS WITH EXPERIENCE
# ================================

cursor.execute("""
INSERT INTO employee_skills (employee_id, skill_id, years_of_experience)
VALUES
(1, 1, 3),
(1, 2, 2),
(1, 3, 1),
(2, 1, 4),
(2, 2, 3)
ON CONFLICT DO NOTHING;
""")

conn.commit()
print("✅ Skill experience mapped")

# ================================
# QUERY DATA
# ================================

print("\n📊 Employee Skills Report\n")

cursor.execute("""
SELECT e.name, s.skill_name, es.years_of_experience
FROM employee_skills es
JOIN employees e ON es.employee_id = e.id
JOIN skills s ON es.skill_id = s.id
ORDER BY e.name;
""")

rows = cursor.fetchall()

for row in rows:
    print(f"Employee: {row[0]}, Skill: {row[1]}, Experience: {row[2]} years")

# ================================
# CLEANUP
# ================================
cursor.close()
conn.close()
print("\n🔌 Connection closed")
