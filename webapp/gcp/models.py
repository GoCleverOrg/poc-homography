from django.db import models


class CameraConfig(models.Model):
    """
    Camera configuration model for the camera_configs table.

    This model stores camera configurations that are populated by the infra
    project's GitHub Actions. The infra workflow fetches camera data from
    inventory, decrypts credentials with Ansible Vault, and writes to this table.
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
        db_table = "camera_configs"

    def __str__(self):
        return f"{self.camera_id} ({self.tenant_id})"
