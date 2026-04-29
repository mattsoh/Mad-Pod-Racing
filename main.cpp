#include <bits/stdc++.h>
using namespace std;

struct Vec {
    double x = 0.0;
    double y = 0.0;
};

static Vec operator+(const Vec& a, const Vec& b) { return {a.x + b.x, a.y + b.y}; }
static Vec operator-(const Vec& a, const Vec& b) { return {a.x - b.x, a.y - b.y}; }
static Vec operator*(const Vec& a, double k) { return {a.x * k, a.y * k}; }

static double dot(const Vec& a, const Vec& b) { return a.x * b.x + a.y * b.y; }
static double norm2(const Vec& a) { return dot(a, a); }
static double norm(const Vec& a) { return sqrt(norm2(a)); }

static Vec normalize(const Vec& a) {
    double n = norm(a);
    if (n < 1e-9) return {0.0, 0.0};
    return {a.x / n, a.y / n};
}

static double toDeg(double rad) { return rad * 180.0 / M_PI; }

static double angleDiff(double a, double b) {
    double d = fmod(a - b + 540.0, 360.0) - 180.0;
    return fabs(d);
}

struct PodState {
    Vec pos;
    Vec vel;
    int angle = 0;
    int next = 0;
    int prevNext = -1;
    int lap = 0;
    int shieldCooldown = 0;
};

struct Command {
    int x = 0;
    int y = 0;
    string thrust;
};

static double facingToTargetAngle(const PodState& p, const Vec& target) {
    Vec dir = target - p.pos;
    double targetAngle = toDeg(atan2(dir.y, dir.x));
    return angleDiff(targetAngle, static_cast<double>(p.angle));
}

static int computeThrust(double ang, double dist) {
    int thrust = 100;
    if (ang > 90.0) thrust = 0;
    else if (ang > 60.0) thrust = 30;
    else if (ang > 45.0) thrust = 50;
    else if (ang > 30.0) thrust = 70;

    if (dist < 1200.0) thrust = min(thrust, 60);
    if (dist < 800.0) thrust = min(thrust, 30);
    return thrust;
}

static bool shouldShield(const PodState& self, const PodState& enemy) {
    Vec rel = self.vel - enemy.vel;
    double relSpeed = norm(rel);
    double dist = norm(self.pos - enemy.pos);
    return dist < 800.0 && relSpeed > 250.0;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int laps = 0;
    int checkpointCount = 0;
    if (!(cin >> laps)) return 0;
    cin >> checkpointCount;

    vector<Vec> checkpoints(checkpointCount);
    for (int i = 0; i < checkpointCount; ++i) {
        cin >> checkpoints[i].x >> checkpoints[i].y;
    }

    PodState my[2];
    PodState enemy[2];
    bool boostAvailable = true;

    while (true) {
        for (int i = 0; i < 2; ++i) {
            if (!(cin >> my[i].pos.x >> my[i].pos.y >> my[i].vel.x >> my[i].vel.y >> my[i].angle >> my[i].next)) {
                return 0;
            }
            if (my[i].prevNext != -1 && my[i].prevNext == checkpointCount - 1 && my[i].next == 0) {
                my[i].lap += 1;
            }
            my[i].prevNext = my[i].next;
            if (my[i].shieldCooldown > 0) my[i].shieldCooldown -= 1;
        }
        for (int i = 0; i < 2; ++i) {
            cin >> enemy[i].pos.x >> enemy[i].pos.y >> enemy[i].vel.x >> enemy[i].vel.y >> enemy[i].angle >> enemy[i].next;
            if (enemy[i].prevNext != -1 && enemy[i].prevNext == checkpointCount - 1 && enemy[i].next == 0) {
                enemy[i].lap += 1;
            }
            enemy[i].prevNext = enemy[i].next;
        }

        auto progressScore = [checkpointCount](const PodState& p) {
            return p.lap * checkpointCount + p.next;
        };

        auto distanceToNext = [&](const PodState& p) {
            return norm(checkpoints[p.next] - p.pos);
        };

        int myRacer = 0;
        int myBlocker = 1;
        {
            long long p0 = progressScore(my[0]);
            long long p1 = progressScore(my[1]);
            if (p1 > p0) { myRacer = 1; myBlocker = 0; }
            else if (p1 == p0) {
                if (distanceToNext(my[1]) < distanceToNext(my[0])) {
                    myRacer = 1; myBlocker = 0;
                }
            }
        }

        int enemyRacer = 0;
        {
            long long p0 = progressScore(enemy[0]);
            long long p1 = progressScore(enemy[1]);
            enemyRacer = (p1 > p0) ? 1 : 0;
            if (p1 == p0) {
                if (distanceToNext(enemy[1]) < distanceToNext(enemy[0])) enemyRacer = 1;
            }
        }

        Command out[2];

        // Racer logic
        {
            PodState& p = my[myRacer];
            Vec cp = checkpoints[p.next];
            Vec nextCp = checkpoints[(p.next + 1) % checkpointCount];
            double dist = norm(cp - p.pos);

            Vec target = cp;
            if (dist < 2500.0) {
                Vec exitDir = normalize(nextCp - cp);
                target = cp + exitDir * 800.0;
            }

            double speed = norm(p.vel);
            if (speed > 0.0) {
                Vec velDir = normalize(p.vel);
                target = target - velDir * min(1200.0, speed * 3.0);
            }

            double ang = facingToTargetAngle(p, target);
            int thrust = computeThrust(ang, dist);

            bool useShield = false;
            if (p.shieldCooldown == 0) {
                if (shouldShield(p, enemy[0]) || shouldShield(p, enemy[1])) {
                    useShield = true;
                    p.shieldCooldown = 3;
                }
            }

            if (!useShield && boostAvailable && ang < 5.0 && dist > 5000.0) {
                out[myRacer] = {static_cast<int>(target.x), static_cast<int>(target.y), "BOOST"};
                boostAvailable = false;
            } else if (useShield) {
                out[myRacer] = {static_cast<int>(target.x), static_cast<int>(target.y), "SHIELD"};
            } else {
                out[myRacer] = {static_cast<int>(target.x), static_cast<int>(target.y), to_string(thrust)};
            }
        }

        // Blocker logic
        {
            PodState& p = my[myBlocker];
            PodState& e = enemy[enemyRacer];

            Vec eCp = checkpoints[e.next];
            double eDist = norm(eCp - e.pos);

            Vec target = (eDist > 2000.0) ? eCp : (e.pos + e.vel * 3.0);
            double dist = norm(target - p.pos);
            double ang = facingToTargetAngle(p, target);

            int thrust = computeThrust(ang, dist);
            thrust = min(100, thrust + 20);

            bool useShield = false;
            if (p.shieldCooldown == 0) {
                if (shouldShield(p, enemy[0]) || shouldShield(p, enemy[1])) {
                    useShield = true;
                    p.shieldCooldown = 3;
                }
            }

            if (useShield) {
                out[myBlocker] = {static_cast<int>(target.x), static_cast<int>(target.y), "SHIELD"};
            } else {
                out[myBlocker] = {static_cast<int>(target.x), static_cast<int>(target.y), to_string(thrust)};
            }
        }

        cout << out[0].x << " " << out[0].y << " " << out[0].thrust << "\n";
        cout << out[1].x << " " << out[1].y << " " << out[1].thrust << "\n";
    }

    return 0;
}
