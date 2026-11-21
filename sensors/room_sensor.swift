// room_sensor: what a MacBook can tell you about the room and the person, without a camera or a recording.
//
// Prints one JSON line every N seconds:
//   {"t": 1795280000, "lux": 388.4, "idle": 2.1, "locked": false, "display_asleep": false}
//
//   lux             ambient light, from the built-in light sensor through the private IOHIDEventSystemClient API
//                   (Apple Silicon: usage page 0xff00, usage 4, event type 12). Returns nothing while the display is
//                   off: the sensor stops reporting when the screen sleeps.
//   idle            seconds since the last keyboard/mouse/trackpad input (IOHIDSystem HIDIdleTime)
//   locked          screen lock state (CGSession)
//   display_asleep  main display asleep
//
// Build (the default target of recent toolchains is too new for LaunchServices on some systems):
//   swiftc -O -target arm64-apple-macos14.0 room_sensor.swift -o room-sensor
//   ./room-sensor 30        # one line every 30 seconds
import CoreGraphics
import Foundation
import IOKit

// ---------------- ambient light (private API) ----------------
@_silgen_name("IOHIDEventSystemClientCreate") func IOHIDEventSystemClientCreate(_ a: CFAllocator?) -> Unmanaged<AnyObject>?
@_silgen_name("IOHIDEventSystemClientSetMatching") func IOHIDEventSystemClientSetMatching(_ c: AnyObject, _ m: CFDictionary) -> Int32
@_silgen_name("IOHIDEventSystemClientCopyServices") func IOHIDEventSystemClientCopyServices(_ c: AnyObject) -> Unmanaged<CFArray>?
@_silgen_name("IOHIDServiceClientCopyEvent") func IOHIDServiceClientCopyEvent(_ s: AnyObject, _ type: Int64, _ a: Int32, _ b: Int64) -> Unmanaged<AnyObject>?
@_silgen_name("IOHIDEventGetFloatValue") func IOHIDEventGetFloatValue(_ e: AnyObject, _ field: Int32) -> Double

let hid = IOHIDEventSystemClientCreate(kCFAllocatorDefault)?.takeRetainedValue()
if let hid { _ = IOHIDEventSystemClientSetMatching(hid, ["PrimaryUsagePage": 0xff00, "PrimaryUsage": 4] as CFDictionary) }

func lux() -> Double? {
    guard let hid, let svcs = IOHIDEventSystemClientCopyServices(hid)?.takeRetainedValue() as? [AnyObject] else { return nil }
    for s in svcs {
        if let e = IOHIDServiceClientCopyEvent(s, 12, 0, 0)?.takeRetainedValue() {  // 12 = ambient light event
            return IOHIDEventGetFloatValue(e, Int32(12 << 16))                       // field: event type << 16
        }
    }
    return nil
}

// ---------------- input idle, lock, display ----------------
func idleSeconds() -> Double? {
    let svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("IOHIDSystem"))
    guard svc != 0 else { return nil }
    defer { IOObjectRelease(svc) }
    guard let v = IORegistryEntryCreateCFProperty(svc, "HIDIdleTime" as CFString, kCFAllocatorDefault, 0)?
        .takeRetainedValue() as? NSNumber else { return nil }
    return v.doubleValue / 1e9  // nanoseconds
}

func screenLocked() -> Bool {
    guard let d = CGSessionCopyCurrentDictionary() as? [String: Any] else { return false }
    return (d["CGSSessionScreenIsLocked"] as? Bool) ?? false
}

func sample() -> [String: Any] {
    var rec: [String: Any] = ["t": Int(Date().timeIntervalSince1970), "locked": screenLocked(),
                              "display_asleep": CGDisplayIsAsleep(CGMainDisplayID()) != 0]
    if let l = lux() { rec["lux"] = (l * 10).rounded() / 10 }
    if let i = idleSeconds() { rec["idle"] = (i * 10).rounded() / 10 }
    return rec
}

let every = Double(CommandLine.arguments.dropFirst().first ?? "") ?? 0
repeat {
    if let d = try? JSONSerialization.data(withJSONObject: sample(), options: [.sortedKeys]), let s = String(data: d, encoding: .utf8) {
        print(s)
        fflush(stdout)
    }
    if every > 0 { Thread.sleep(forTimeInterval: every) }
} while every > 0
