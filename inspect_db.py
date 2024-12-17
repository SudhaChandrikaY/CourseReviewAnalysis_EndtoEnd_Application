import sqlite3

DATABASE = "reviews.db"

def print_reviews():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT * FROM reviews")
    rows = c.fetchall()
    conn.close()

    if rows:
        print("Reviews Table:")
        for row in rows:
            print(row)
    else:
        print("No reviews found in the database.")
        

def update_sentiments_to_numeric():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # Update text sentiments to numeric values
    c.execute("UPDATE reviews SET sentiment = 5 WHERE sentiment = 'very good'")
    c.execute("UPDATE reviews SET sentiment = 4 WHERE sentiment = 'good'")
    c.execute("UPDATE reviews SET sentiment = 3 WHERE sentiment = 'average'")
    c.execute("UPDATE reviews SET sentiment = 2 WHERE sentiment = 'poor'")
    c.execute("UPDATE reviews SET sentiment = 1 WHERE sentiment = 'very poor'")

    conn.commit()
    conn.close()
    print("Updated sentiments to numeric values.")

if __name__ == "__main__":
    print_reviews()
    update_sentiments_to_numeric()