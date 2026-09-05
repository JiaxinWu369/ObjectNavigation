from ai2thor.controller import Controller


controller = Controller()

controller.start()

controller.reset(
    "FloorPlan1"
)

event = controller.step(
    dict(
        action="GetReachablePositions"
    )
)

print("=" * 80)
print("GET REACHABLE POSITIONS")
print("=" * 80)

print(
    "lastActionSuccess:",
    event.metadata.get(
        "lastActionSuccess"
    )
)

print(
    "errorMessage:",
    event.metadata.get(
        "errorMessage"
    )
)

positions = (
    event.metadata
    .get(
        "actionReturn",
        []
    )
)

print(
    "num positions:",
    len(positions)
)

if positions:
    print(
        "first position:",
        positions[0]
    )


print()
print("=" * 80)
print("OBJECT METADATA SAMPLE")
print("=" * 80)

objects = (
    event.metadata.get(
        "objects",
        []
    )
)

print(
    "num objects:",
    len(objects)
)

for obj in objects[:5]:

    print(
        {
            "objectType":
                obj.get(
                    "objectType"
                ),

            "objectId":
                obj.get(
                    "objectId"
                ),

            "visible":
                obj.get(
                    "visible"
                ),

            "position":
                obj.get(
                    "position"
                ),
        }
    )


if positions:

    p = positions[0]

    print()
    print("=" * 80)
    print("TELEPORT VISIBILITY PROBE")
    print("=" * 80)

    for yaw in [
        0,
        90,
        180,
        270,
    ]:

        for horizon in [
            -30,
            0,
            30,
        ]:

            ev = controller.step(
                dict(
                    action="TeleportFull",
                    x=float(
                        p["x"]
                    ),
                    y=float(
                        p["y"]
                    ),
                    z=float(
                        p["z"]
                    ),
                    rotation=float(
                        yaw
                    ),
                    horizon=float(
                        horizon
                    ),
                )
            )

            success = (
                ev.metadata.get(
                    "lastActionSuccess"
                )
            )

            visible = sorted(
                set(
                    obj[
                        "objectType"
                    ]
                    for obj in
                    ev.metadata.get(
                        "objects",
                        []
                    )
                    if obj.get(
                        "visible",
                        False,
                    )
                )
            )

            print(
                "yaw={:<3} "
                "horizon={:<3} "
                "success={} "
                "visible={}".format(
                    yaw,
                    horizon,
                    success,
                    visible,
                )
            )


controller.stop()
