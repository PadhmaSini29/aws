import psycopg2
from typing import Dict, Any


def run_postgres_query(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    sql: str
) -> Dict[str, Any]:

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=database,
        user=user,
        password=password
    )

    try:
        cursor = conn.cursor()
        cursor.execute(sql)

        if cursor.description:
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            return {"columns": columns, "rows": rows}
        else:
            conn.commit()
            return {"columns": [], "rows": []}

    finally:
        cursor.close()
        conn.close()
