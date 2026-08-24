import os

import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '123456789')
DB_NAME = os.getenv('DB_NAME', 'seek_crawler')


def create_connection():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        if connection.is_connected():
            print("Connected")

        cursor = connection.cursor()
        return connection, cursor

    except mysql.connector.Error as error:
        print("Error en conexion a BD:", error)
        return None, None


def close_connection(connection):
    if connection and connection.is_connected():
        connection.close()
        print("Conexión cerrada")


def get_tickets():
    connection, cursor = create_connection()
    if not connection:
        return []
    try:
        cursor.execute(
            "SELECT id, categoria, asunto, consulta, resolucion, prioridad, fecha FROM tickets"
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            ticket_id, categoria, asunto, consulta, resolucion, prioridad, fecha = row
            results.append(
                "FUENTE: TICKET-%s\n"
                "Categoría: %s | Asunto: %s | Prioridad: %s | Fecha: %s\n"
                "Consulta: %s\n"
                "Resolución: %s" % (
                    ticket_id, categoria, asunto, prioridad, fecha, consulta, resolucion
                )
            )
        return results
    except mysql.connector.Error as error:
        print("Error al obtener tickets:", error)
        return []
    finally:
        close_connection(connection)


def get_sop():
    connection, cursor = create_connection()
    if not connection:
        return []
    try:
        cursor.execute("SELECT codigo, titulo, area, procedimiento FROM sop")
        rows = cursor.fetchall()
        results = []
        for codigo, titulo, area, procedimiento in rows:
            results.append(
                "FUENTE: %s\n"
                "Título: %s | Área: %s\n"
                "Procedimiento:\n%s" % (codigo, titulo, area, procedimiento)
            )
        return results
    except mysql.connector.Error as error:
        print("Error al obtener SOP:", error)
        return []
    finally:
        close_connection(connection)


def getdata():
    return get_tickets() + get_sop()


def count_rows(table):
    connection, cursor = create_connection()
    if not connection:
        return 0
    try:
        cursor.execute("SELECT COUNT(*) FROM {}".format(table))
        return cursor.fetchone()[0]
    except mysql.connector.Error as error:
        print("Error al contar {}:".format(table), error)
        return 0
    finally:
        close_connection(connection)


def run_seed_file():
    connection, cursor = create_connection()
    if not connection:
        raise RuntimeError("No se pudo conectar a MySQL para cargar el seed.sql")
    try:
        with open('seed.sql', 'r', encoding='utf-8') as f:
            sql = f.read()
        for _ in cursor.execute(sql, multi=True):
            pass
        connection.commit()
        print("seed.sql aplicado correctamente")
    except mysql.connector.Error as error:
        connection.rollback()
        raise RuntimeError("Error ejecutando seed.sql: {}".format(error))
    finally:
        close_connection(connection)
