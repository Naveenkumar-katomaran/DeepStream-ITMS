import pika
import json
import logging as log
logging = log.getLogger(__name__)
log.getLogger("pika").setLevel(log.WARNING)

class RabbitMQProducer:
    def __init__(self, config):
        self.config = config
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 5672)
        self.username = config.get("username", "guest")
        self.password = config.get("password", "guest")
        self.queue_name = config.get("queue_name", "vehicle_entries")
        
        self.connection = None
        self.channel = None
        self._connect()

    def _connect(self):
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
                socket_timeout=3,
                connection_attempts=1
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            # Declare queue (Persistence=No as requested)
            self.channel.queue_declare(queue=self.queue_name, durable=False)
            logging.info(f"Connected to RabbitMQ at {self.host}:{self.port}, Queue: {self.queue_name}")
        except Exception as e:
            logging.error(f"Failed to connect to RabbitMQ: {e}")
            self.connection = None
            self.channel = None

    def publish(self, message):
        """Publishes a JSON message to the queue."""
        if not self.channel or self.connection.is_closed:
            logging.warning("RabbitMQ connection lost. Retrying...")
            self._connect()

        if self.channel:
            try:
                body = json.dumps(message)
                self.channel.basic_publish(
                    exchange='',
                    routing_key=self.queue_name,
                    body=body,
                    properties=pika.BasicProperties(
                        content_type='application/json',
                        delivery_mode=1  # 1 = Non-persistent, 2 = Persistent
                    )
                )
                return True
            except Exception as e:
                logging.error(f"Failed to publish to RabbitMQ: {e}")
                return False
        return False

    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()
