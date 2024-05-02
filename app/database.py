import mysql.connector

def fetch_craftsmen_from_database(host, user, password, database, port):
    craftsmen_data = []
    try:
        # Connect to the database
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )

        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            # Execute query to fetch craftsmen details
            cursor.execute("SELECT user.photo,user.f_name,user.l_name,user.Longitude,user.latitude,user.address,craftsman.category,craftsman.price,craftsman.rate FROM user JOIN craftsman ON user.id = craftsman.userid;")

            # Fetch all rows

            # Fetch all rows
            rows = cursor.fetchall()

            # Append craftsmen details to the list
            for row in rows:
                craftsmen_data.append(row)

    except mysql.connector.Error as e:
        print("Error connecting to the database:", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

    return craftsmen_data
