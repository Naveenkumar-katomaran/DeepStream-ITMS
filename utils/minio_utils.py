import os
import logging as log
from minio import Minio
from minio.error import S3Error
import mimetypes
import io

logging = log.getLogger(__name__)

class MinioClient:
    def __init__(self, config):
        self.endpoint = config.get("host", "127.0.0.1:9000").replace("http://", "").replace("https://", "")
        self.access_key = config.get("username", "minioadmin")
        self.secret_key = config.get("password", "minioadmin")
        self.bucket = config.get("bucket", "traffic-api")
        self.secure = config.get("secure", False)
        self.expire_seconds = config.get("expire_seconds", 3600 * 24 * 7)
        self.public_url = config.get("public_url", "").rstrip("/")
        
        protocol = "https" if self.secure else "http"
        self.endpoint_url = f"{protocol}://{self.endpoint}"
        
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        try:
            self.ensure_bucket_exists()
        except Exception as e:
            logging.error(f"MinIO initialization error: {e}")

    def ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logging.info(f"Created MinIO bucket: {self.bucket}")
        except Exception as e:
            logging.error(f"Error checking/creating bucket: {e}")
            raise

    def upload_bytes(self, data, object_key, content_type=None):
        """Upload bytes directly from memory."""
        try:
            if content_type is None:
                content_type, _ = mimetypes.guess_type(object_key)
            
            if object_key.lower().endswith(".mp4") and (content_type is None or content_type == "application/octet-stream"):
                content_type = "video/mp4"
            if content_type is None:
                content_type = "image/jpeg"

            data_stream = io.BytesIO(data)
            self.client.put_object(
                bucket_name=self.bucket,
                object_name=object_key,
                data=data_stream,
                length=len(data),
                content_type=content_type
            )
            logging.info(f"Uploaded (bytes) to MinIO -> {self.bucket}/{object_key} ({content_type})")
            return self.get_public_url(object_key)
        except Exception as exc:
            logging.error(f"MinIO byte-upload failed: {self.bucket}/{object_key} - {exc}")
            return None

    def upload_file(self, file_path, object_key, content_type=None):
        """Upload a local file directly using fput_object."""
        try:
            if content_type is None:
                content_type, _ = mimetypes.guess_type(file_path)
            
            if file_path.lower().endswith(".mp4") and (content_type is None or content_type == "application/octet-stream"):
                content_type = "video/mp4"

            self.client.fput_object(
                bucket_name=self.bucket,
                object_name=object_key,
                file_path=file_path,
                content_type=content_type
            )
            logging.info(f"Uploaded (file) to MinIO -> {self.bucket}/{object_key} ({content_type})")
            return self.get_public_url(object_key)
        except Exception as exc:
            logging.error(f"MinIO file-upload failed: {file_path} -> {object_key} - {exc}")
            return None

    def get_public_url(self, object_key):
        """Returns direct public HTTP URL."""
        base = self.public_url if self.public_url else self.endpoint_url
        return f"{base}/{self.bucket}/{object_key}"

    def get_url(self, object_key, presign=True):
        """Returns HTTP URL (Presigned by default)."""
        if presign:
            try:
                url = self.client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": self.bucket, "Key": object_key},
                    ExpiresIn=self.expire_seconds,
                )
                return url
            except Exception as e:
                logging.warning(f"Presigned URL failed for {object_key}: {e}")
        
        return self.get_public_url(object_key)
