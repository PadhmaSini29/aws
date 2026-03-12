import sqlite3
from typing import Dict, Any, List

def run_sqlite_query(db_path: str, sql: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        if cur.description:
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
            return {"columns": cols, "rows": rows}
        else:
            conn.commit()
            return {"columns": [], "rows": [], "message": "OK"}
    finally:
        conn.close()
