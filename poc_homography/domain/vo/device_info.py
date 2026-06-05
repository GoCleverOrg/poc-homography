"""Device information value object (canonical, DDD)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from poc_homography.domain.vo._xml import find_text

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element


@dataclass(frozen=True)
class DeviceInfo:
    """Identity and firmware metadata for a camera device.

    Attributes:
        model: Hardware model string (e.g., ``DS-2DF8425IX-AELW``).
        serial_number: Full serial number.
        mac_address: MAC address.
        firmware_version: Firmware version string.
        device_name: User-assigned device name.
        device_type: Device type (e.g., ``IPDome``).
        device_id: Globally unique device identifier.
        device_description: Free-form device description.
        encoder_version: Encoder firmware version.
        boot_version: Boot loader version.
        hardware_version: Hardware revision string.
        platform_name: Platform/SoC name (e.g., ``H8``).
        manufacturer: Manufacturer name.
    """

    model: str | None
    serial_number: str | None
    mac_address: str | None
    firmware_version: str | None
    device_name: str | None
    device_type: str | None
    device_id: str | None
    device_description: str | None
    encoder_version: str | None
    boot_version: str | None
    hardware_version: str | None
    platform_name: str | None
    manufacturer: str | None

    @classmethod
    def from_element(cls, elem: Element) -> DeviceInfo:
        """Build :class:`DeviceInfo` from a ``DeviceInfo`` element.

        Args:
            elem: The ISAPI ``DeviceInfo`` root element.

        Returns:
            A populated :class:`DeviceInfo`.
        """
        return cls(
            model=find_text(elem, "model"),
            serial_number=find_text(elem, "serialNumber"),
            mac_address=find_text(elem, "macAddress"),
            firmware_version=find_text(elem, "firmwareVersion"),
            device_name=find_text(elem, "deviceName"),
            device_type=find_text(elem, "deviceType"),
            device_id=find_text(elem, "deviceID"),
            device_description=find_text(elem, "deviceDescription"),
            encoder_version=find_text(elem, "encoderVersion"),
            boot_version=find_text(elem, "bootVersion"),
            hardware_version=find_text(elem, "hardwareVersion"),
            platform_name=find_text(elem, "platformName"),
            manufacturer=find_text(elem, "manufacturer"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "serial_number": self.serial_number,
            "mac_address": self.mac_address,
            "firmware_version": self.firmware_version,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "device_id": self.device_id,
            "device_description": self.device_description,
            "encoder_version": self.encoder_version,
            "boot_version": self.boot_version,
            "hardware_version": self.hardware_version,
            "platform_name": self.platform_name,
            "manufacturer": self.manufacturer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceInfo:
        """Create :class:`DeviceInfo` from a dictionary."""
        return cls(
            model=data.get("model"),
            serial_number=data.get("serial_number"),
            mac_address=data.get("mac_address"),
            firmware_version=data.get("firmware_version"),
            device_name=data.get("device_name"),
            device_type=data.get("device_type"),
            device_id=data.get("device_id"),
            device_description=data.get("device_description"),
            encoder_version=data.get("encoder_version"),
            boot_version=data.get("boot_version"),
            hardware_version=data.get("hardware_version"),
            platform_name=data.get("platform_name"),
            manufacturer=data.get("manufacturer"),
        )
