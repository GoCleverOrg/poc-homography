from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="CameraConfig",
            fields=[
                (
                    "camera_id",
                    models.CharField(max_length=100, primary_key=True, serialize=False),
                ),
                ("tenant_id", models.CharField(max_length=50)),
                (
                    "camera_ip",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                ("username", models.CharField(max_length=200)),
                ("password", models.CharField(max_length=200)),
                (
                    "model",
                    models.CharField(blank=True, max_length=100, null=True),
                ),
                (
                    "vendor",
                    models.CharField(blank=True, max_length=100, null=True),
                ),
                ("subtype", models.CharField(max_length=100)),
                (
                    "serial_number",
                    models.CharField(blank=True, max_length=100, null=True),
                ),
                (
                    "mac_address",
                    models.CharField(blank=True, max_length=100, null=True),
                ),
                (
                    "firmware_version",
                    models.CharField(blank=True, max_length=100, null=True),
                ),
                (
                    "location",
                    models.CharField(blank=True, max_length=100, null=True),
                ),
                (
                    "status",
                    models.CharField(blank=True, max_length=100, null=True),
                ),
            ],
            options={
                "db_table": "camera_configs",
            },
        ),
    ]
