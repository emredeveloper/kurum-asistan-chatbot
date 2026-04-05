import datetime
import os
import sqlite3
from typing import Dict, List, Optional


def get_db_name():
    return os.environ.get('TEST_DATABASE_URL', 'chatbot_data.db')


def get_db_connection():
    conn = sqlite3.connect(get_db_name())
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_name_override=None):
    current_db_name = db_name_override or get_db_name()
    conn = sqlite3.connect(current_db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            type TEXT,
            user_message TEXT,
            bot_response TEXT,
            details TEXT
        )
        '''
    )

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT,
            title TEXT,
            department TEXT
        )
        '''
    )

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS support_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            ticket_id TEXT UNIQUE NOT NULL,
            department TEXT,
            description TEXT,
            status TEXT DEFAULT 'open',
            priority TEXT DEFAULT 'normal',
            category TEXT DEFAULT 'general',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS uploaded_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            stored_filename TEXT UNIQUE NOT NULL,
            uploader_name TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            processed INTEGER DEFAULT 0
        )
        '''
    )

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS institution_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keywords TEXT NOT NULL,
            answer TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    conn.commit()
    seed_default_knowledge(conn)
    seed_default_users(conn)
    conn.close()


def seed_default_users(conn: Optional[sqlite3.Connection] = None):
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM users')
    count = cur.fetchone()[0]
    if count and count > 0:
        if close_conn:
            conn.close()
        return

    default_users = [
        ("u-it-01", "Ali Yilmaz", "Senior Software Engineer", "IT"),
        ("u-hr-01", "Ayse Demir", "HR Specialist", "Human Resources"),
        ("u-fin-01", "Mehmet Kaya", "Accounting Specialist", "Accounting"),
        ("u-tech-01", "Zeynep Koc", "Technical Service Lead", "Technical Service"),
        ("u-ops-01", "Can Arslan", "Operations Manager", "Operations"),
    ]
    cur.executemany('INSERT OR IGNORE INTO users (id, name, title, department) VALUES (?, ?, ?, ?)', default_users)
    conn.commit()

    if close_conn:
        conn.close()


def seed_default_knowledge(conn: Optional[sqlite3.Connection] = None):
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    cur = conn.cursor()
    defaults = [
        (
            "travel policy, travel, trip, accommodation",
            "Out-of-town business trips require prior approval and all expenses must be documented with invoices.",
        ),
        (
            "leave procedure, leave, annual leave",
            "Annual leave requests should be submitted to Human Resources at least 3 days in advance.",
        ),
        (
            "overtime pay, overtime",
            "Overtime payments are reflected in payroll at the end of the relevant month.",
        ),
        (
            "cafeteria, lunch, meal",
            "The cafeteria is open on weekdays between 12:00 and 14:00.",
        ),
    ]

    for keywords, answer in defaults:
        cur.execute('SELECT id, answer FROM institution_knowledge WHERE keywords = ?', (keywords,))
        row = cur.fetchone()
        if row:
            if row['answer'] != answer:
                cur.execute(
                    "UPDATE institution_knowledge SET answer = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (answer, row['id']),
                )
        else:
            cur.execute('INSERT INTO institution_knowledge (keywords, answer) VALUES (?, ?)', (keywords, answer))

    conn.commit()

    if close_conn:
        conn.close()


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def search_kb_answer(query: str) -> Optional[str]:
    if not query or not query.strip():
        return None
    q = _normalize_text(query)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT keywords, answer FROM institution_knowledge')
    rows = cur.fetchall()
    conn.close()
    for row in rows:
        keywords = (row[0] or "").split(',')
        for kw in keywords:
            if not kw.strip():
                continue
            if _normalize_text(kw) in q:
                return row[1]
    return None


def search_kb_entries(query: str) -> List[Dict]:
    results: List[Dict] = []
    if not query or not query.strip():
        return results
    q = _normalize_text(query)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT keywords, answer FROM institution_knowledge')
    rows = cur.fetchall()
    conn.close()
    for row in rows:
        keywords = (row[0] or "").split(',')
        for kw in keywords:
            if not kw.strip():
                continue
            if _normalize_text(kw) in q:
                results.append({"keywords": row[0], "answer": row[1]})
                break
    return results


def get_users():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, name, title, department FROM users ORDER BY name ASC')
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_departments() -> List[str]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT DISTINCT department FROM users ORDER BY department ASC')
    departments = [row[0] for row in cur.fetchall()]
    conn.close()
    return departments


def add_chat_history(user_id: str, type: str, user_message: str, bot_response: str, details_json_str: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (user_id, type, user_message, bot_response, details) VALUES (?, ?, ?, ?, ?)",
        (user_id, type, user_message, bot_response, details_json_str),
    )
    conn.commit()
    conn.close()


def get_chat_history(user_id: str, limit: int = 20) -> list:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit),
    )
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return history


def add_support_ticket(user_id: str, ticket_id: str, department: str, description: str, priority: str, category: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now()
    cursor.execute(
        "INSERT INTO support_tickets (user_id, ticket_id, department, description, priority, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, ticket_id, department, description, priority, category, now, now),
    )
    conn.commit()
    conn.close()


def get_support_tickets(user_id: str, limit: int = 50) -> list:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM support_tickets WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    )
    tickets = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return tickets


def get_support_tickets_all(limit: int = 100) -> list:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT st.*, u.name as user_name, u.department as user_department
        FROM support_tickets st
        LEFT JOIN users u ON u.id = st.user_id
        ORDER BY st.created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    tickets = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return tickets


def update_support_ticket_status(user_id: str, ticket_id: str, status: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now()
    cursor.execute(
        "UPDATE support_tickets SET status = ?, updated_at = ? WHERE user_id = ? AND ticket_id = ?",
        (status, now, user_id, ticket_id),
    )
    updated_rows = cursor.rowcount
    conn.commit()
    conn.close()
    return updated_rows > 0


def get_ticket_by_id(user_id: str, ticket_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM support_tickets WHERE user_id = ? AND ticket_id = ?",
        (user_id, ticket_id),
    )
    ticket = cursor.fetchone()
    conn.close()
    return dict(ticket) if ticket else None


def add_report(user_id: str, original_filename: str, stored_filename: str, uploader_name: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO uploaded_reports (user_id, original_filename, stored_filename, uploader_name) VALUES (?, ?, ?, ?)",
        (user_id, original_filename, stored_filename, uploader_name),
    )
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return report_id


def get_reports(user_id: str = None, limit: int = 100) -> list:
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id:
        cursor.execute(
            "SELECT * FROM uploaded_reports WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
    else:
        cursor.execute(
            "SELECT * FROM uploaded_reports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
    reports = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return reports


def get_report_by_id(report_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM uploaded_reports WHERE id = ?", (report_id,))
    report = cursor.fetchone()
    conn.close()
    return dict(report) if report else None


def mark_report_as_processed(report_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE uploaded_reports SET processed = 1 WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()


def get_unprocessed_reports() -> list:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM uploaded_reports WHERE processed = 0 ORDER BY created_at ASC")
    reports = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return reports


def delete_report(report_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT stored_filename FROM uploaded_reports WHERE id = ?", (report_id,))
    report = cursor.fetchone()
    if not report:
        conn.close()
        return None

    stored_filename = report['stored_filename']
    cursor.execute("DELETE FROM uploaded_reports WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()
    return stored_filename


def delete_all_reports():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM uploaded_reports')
    conn.commit()
    conn.close()


def reset_non_user_data():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('DELETE FROM chat_history')
    cur.execute('DELETE FROM support_tickets')
    cur.execute('DELETE FROM uploaded_reports')
    cur.execute('DELETE FROM institution_knowledge')
    conn.commit()
    conn.close()


if __name__ == '__main__':
    print("Initializing database...")
    init_db()
    print("Database initialized.")
