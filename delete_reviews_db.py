import sqlite3

DATABASE = "reviews.db"

def delete_unwanted_reviews():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # List of IDs to delete
    ids_to_delete = [1,2,3,4,5,7,8,9,10]

    # Delete rows with these IDs
    for review_id in ids_to_delete:
        c.execute("DELETE FROM reviews WHERE id = ?", (review_id,))

    conn.commit()
    conn.close()
    print(f"Deleted rows with IDs: {ids_to_delete}")

if __name__ == "__main__":
    delete_unwanted_reviews()
