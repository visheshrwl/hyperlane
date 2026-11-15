import socket

from zeroconf import ServiceInfo, ServiceStateChange

from shardline.discovery import SERVICE_TYPE, DiscoveryManager, parse_service_info

# Mirrors build_txt_record() in worker/src/discovery.cc.
WORKER_TXT = {
    "txtvers": "1",
    "proto": "shardline.v1",
    "id": "mac-mini-1",
    "devices": "2",
    "kind": "metal",
    "device": "Apple M5",
    "mem_total": "17179869184",
    "mem_usable": "12713115648",
}


LAN_ADDRESS = socket.inet_aton("192.168.1.20")


def service_info(properties, addresses=(LAN_ADDRESS,), port=50051):
    return ServiceInfo(
        SERVICE_TYPE,
        f"mac-mini-1.{SERVICE_TYPE}",
        port=port,
        properties=properties,
        addresses=list(addresses),
    )


def test_parses_one_txt_key_per_field():
    advert = parse_service_info(service_info(WORKER_TXT))

    assert advert is not None
    assert advert.worker_id == "mac-mini-1"
    assert advert.protocol == "shardline.v1"
    assert advert.device_kind == "metal"
    assert advert.device_name == "Apple M5"
    assert advert.total_memory_bytes == 17179869184
    assert advert.usable_memory_bytes == 12713115648
    assert advert.target == "192.168.1.20:50051"


def test_ipv6_target_is_bracketed():
    address = socket.inet_pton(socket.AF_INET6, "fd00::20")
    advert = parse_service_info(service_info(WORKER_TXT, addresses=(address,)))
    assert advert.target == "[fd00::20]:50051"


def test_rejects_unknown_txt_version():
    assert parse_service_info(service_info({**WORKER_TXT, "txtvers": "2"})) is None


def test_malformed_memory_reads_as_zero():
    advert = parse_service_info(service_info({**WORKER_TXT, "mem_usable": "lots"}))
    assert advert.usable_memory_bytes == 0


def test_no_sockets_until_started():
    created = []
    manager = DiscoveryManager(zeroconf_factory=lambda: created.append(1))
    assert created == []
    assert manager.workers() == []


def test_removed_service_is_forgotten():
    manager = DiscoveryManager()
    advert = parse_service_info(service_info(WORKER_TXT))
    manager._workers[advert.name] = advert

    manager._on_change(None, SERVICE_TYPE, advert.name, ServiceStateChange.Removed)

    assert manager.workers() == []
