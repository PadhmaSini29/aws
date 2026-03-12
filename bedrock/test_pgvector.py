from sqlalchemy import create_engine, text

CONN = "postgresql://postgres:NewPassword123!@localhost:5432/bedrock_rag"

engine = create_engine(CONN)

try:
    # Test DB connection
    with engine.connect() as conn:
        print("✅ Connected to PostgreSQL!")

        # Test pgvector extension
        result = conn.execute(text("SELECT extname FROM pg_extension;"))
        extensions = [row[0] for row in result]

        if "vector" in extensions:
            print("✅ pgvector extension is installed!")
        else:
            print("❌ pgvector NOT installed.")
            print("Run in pgAdmin → CREATE EXTENSION vector;")
            exit()

        # Create test table with vector column
        conn.execute(text("""
            DROP TABLE IF EXISTS test_vectors;
            CREATE TABLE test_vectors (
                id serial PRIMARY KEY,
                embedding vector(3)
            );
        """))
        print("✅ Test table created!")

        # Insert sample vector
        conn.execute(text("""
            INSERT INTO test_vectors (embedding)
            VALUES ('[1,2,3]'), ('[0.5,0.2,0.1]');
        """))
        print("✅ Inserted sample vectors!")

        # Run similarity search
        result = conn.execute(text("""
            SELECT id, embedding <-> '[1,2,2]' AS distance
            FROM test_vectors
            ORDER BY distance ASC
            LIMIT 1;
        """))

        row = result.fetchone()
        print("🔍 Closest vector to [1,2,2]:")
        print(row)

except Exception as e:
    print("❌ ERROR:", e)
