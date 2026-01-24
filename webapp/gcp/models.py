from django.db import models


class CameraConfig(models.Model):
    """
    Camera configuration model for externally-managed camera_configs table.

    This model provides read-only access to camera configurations populated by
    the infra project's GitHub Actions. The table schema is owned by infra,
    so this model uses managed=False to prevent Django from modifying the schema.
    """

    camera_id = models.CharField(max_length=100, primary_key=True)
    tenant_id = models.CharField(max_length=50)
    camera_ip = models.CharField(max_length=50, null=True, blank=True)
    username = models.CharField(max_length=200)
    password = models.CharField(max_length=200)
    model = models.CharField(max_length=100, null=True, blank=True)
    vendor = models.CharField(max_length=100, null=True, blank=True)
    subtype = models.CharField(max_length=100)
    serial_number = models.CharField(max_length=100, null=True, blank=True)
    mac_address = models.CharField(max_length=100, null=True, blank=True)
    firmware_version = models.CharField(max_length=100, null=True, blank=True)
    location = models.CharField(max_length=100, null=True, blank=True)
    status = models.CharField(max_length=100, null=True, blank=True)

    class Meta:
        managed = False
        db_table = "camera_configs"

    def __str__(self):
        return f"{self.camera_id} ({self.tenant_id})"
