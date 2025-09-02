const { Kafka } = require('kafkajs');

const kafka = new Kafka({
  clientId: 'my-app',
  brokers: ['localhost:9092'] // matches your docker-compose
});

const producer = kafka.producer();

const run = async () => {
  await producer.connect();

  // send some test messages
  await producer.send({
    topic: 'ml-data',
    messages: [
      { value: 'First message' },
      { value: 'Second message' },
      { value: 'Third message' },
    ],
  });

  console.log('✅ Messages sent successfully');
  await producer.disconnect();
};

run().catch(console.error);
