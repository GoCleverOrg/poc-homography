from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ


class DummyResponse:
    def __init__(self, status_code=200, text="OK"):
        self.status_code = status_code
        self.text = text


def test_send_ptz_return_sends_expected_xml(monkeypatch):
    captured = {}

    def fake_put(url, auth=None, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["auth"] = auth
        return DummyResponse(200, "OK")

    monkeypatch.setattr("requests.put", fake_put)

    cam = HikvisionPTZ(ip="192.0.2.1", username="u", password="p")
    status = {"pan": 10.5, "tilt": -2.0, "zoom": 3.2}

    ok = cam.send_ptz_return(status)

    assert ok is True
    assert captured["url"].endswith("/ISAPI/PTZCtrl/channels/1/absolute")
    # check that values were multiplied by 10 and present in XML
    assert "<azimuth>105</azimuth>" in captured["data"]
    assert "<elevation>-20</elevation>" in captured["data"]
    assert "<absoluteZoom>32</absoluteZoom>" in captured["data"]
    assert captured["headers"]["Content-Type"] == "application/xml"


def test_send_3d_zoom_command_sends_expected_xml(monkeypatch):
    captured = {}

    def fake_put(url, auth=None, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        return DummyResponse(200, "OK")

    monkeypatch.setattr("requests.put", fake_put)

    cam = HikvisionPTZ(ip="192.0.2.2", username="u", password="p")

    code, resp = cam.send_3d_zoom_command(0.1, 0.2, 0.3, 0.4)

    assert code == 200
    assert captured["url"].endswith("/ISAPI/PTZCtrl/channels/1/position3D")
    assert "<positionX>0.1</positionX>" in captured["data"]
    assert "<positionY>0.2</positionY>" in captured["data"]
    assert "<positionX>0.3</positionX>" in captured["data"]
    assert "<positionY>0.4</positionY>" in captured["data"]
