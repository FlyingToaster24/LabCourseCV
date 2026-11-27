# Poster section: Application scenario (draft, Alper)

## Onboard scene tagging for Earth observation

Small Earth-observation satellites and high-altitude UAVs capture far more image tiles than their downlink can carry.
If the platform classifies every tile on board, it can send the relevant scenes first (airports, harbours, ships,
storage tanks) and hold back tiles of little value, such as cloud or open sea.

We train a land-use scene classifier on NWPU-RESISC45 (31,500 aerial tiles, 45 classes, 0.2 to 30 m per pixel,
over 100 countries) and compress it until it runs in real time on the course's edge hardware.

## Mission requirements

| Requirement | Target |
| --- | --- |
| Model size | 20 MB or less (course limit); 5 MB or less as INT8 for the flight model |
| Accuracy | 90 % or more top-1 on the NWPU-RESISC45 test set |
| Latency | Real time per tile on the provided hardware (value fixed once the hardware is known) |
| Robustness | Small accuracy drop under sensor noise, motion blur, haze, downlink JPEG compression and imagery from another region (UC Merced) |
