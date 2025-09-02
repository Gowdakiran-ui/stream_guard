from confluent_kafka import Consumer, KafkaException
import psycopg2
import json

# --------- PostgreSQL Connection ----------
conn = psycopg2.connect(
    host="localhost:9000",      
    database="fraud_db",    # your DB name
    user="postgres",        # your username
    password="yourpassword" # your password
)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50),
    amount FLOAT,
    timestamp TIMESTAMP,
    device_id VARCHAR(50),
    merchant_id VARCHAR(50),
    is_fraud INT
);
""")
conn.commit()

# --------- Kafka Consumer Config ----------
conf = {
    'bootstrap.servers': 'localhost:9092', # Kafka broker
    'group.id': 'fraud-consumer-group',
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(conf)
consumer.subscribe(['fraud-transactions'])  # your Kafka topic

# --------- Consume & Insert into Postgres ----------
print("Listening for messages...")

try:
    while True:
        msg = consumer.poll(1.0)  # wait 1 second
        if msg is None:
            continue
        if msg.error():
            raise KafkaException(msg.error())

        # Decode message (JSON assumed)
        record = json.loads(msg.value().decode('utf-8'))

        cursor.execute("""
            INSERT INTO transactions (transaction_id, user_id, amount, timestamp, device_id, merchant_id, is_fraud)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (transaction_id) DO NOTHING;
        """, (
            record.get("transaction_id"),
            record.get("user_id"),
            record.get("amount"),
            record.get("timestamp"),
            record.get("device_id"),
            record.get("merchant_id"),
            record.get("is_fraud", 0)  # default 0
        ))
        conn.commit()
        print(f"Inserted transaction {record.get('transaction_id')}")

except KeyboardInterrupt:
    pass
finally:
    consumer.close()
    cursor.close()
    conn.close()
