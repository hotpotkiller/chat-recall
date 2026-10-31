#!/usr/bin/env python3
"""Deterministic builder for ordinary-life passage evaluation benchmark fixture.

Constructs 12 ordinary-life ChatGPT conversations and 64 queries (48 positive,
16 unrelated) with whole-paragraph gold targets, explicit expected_text, and
mechanically derived codepoint offsets. Validates constraints before writing
fixtures/passage-evaluation.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_passages import FixtureValidationError, validate_suite


def build_suite() -> dict:
    conversations = [
        # Calibration 1: Tomato soup sourness vs bread proofing
        {
            "source_id": "chatgpt",
            "account_id": "default",
            "id": "conv-calib-1",
            "title": "Tomato soup acidity balance and winter dough proofing",
            "messages": [
                {
                    "id": "c1-m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "How can I reduce the sharp sourness and acidity in my homemade tomato soup without dumping in spoons of white sugar?"
                },
                {
                    "id": "c1-m2",
                    "parent_id": "c1-m1",
                    "role": "assistant",
                    "text": "Roasting the tomatoes in the oven caramelizes their natural sugars and softens harsh raw acidity before blending. When simmering the soup, grating in a fresh sweet carrot adds subtle sweetness, and stirring in a splash of heavy cream or a quarter teaspoon of baking soda neutralizes the sharp acidic taste without masking the tomato flavor."
                },
                {
                    "id": "c1-m3",
                    "parent_id": "c1-m2",
                    "role": "user",
                    "text": "My bread dough takes forever to rise during winter because our kitchen is cold and drafty. How can I speed up proofing safely?"
                },
                {
                    "id": "c1-m4",
                    "parent_id": "c1-m3",
                    "role": "assistant",
                    "text": "To create an ideal warm fermentation environment without buying a dedicated proofing box, place your covered dough bowl inside an unheated oven with only the interior oven light turned on. The low-wattage incandescent bulb gently warms the sealed oven cavity to approximately eighty degrees Fahrenheit, providing ideal ambient temperature and consistent humidity for steady dough expansion."
                }
            ]
        },

        # Calibration 2: Hotel exception refund vs flight missed connection
        {
            "source_id": "chatgpt",
            "account_id": "default",
            "id": "conv-calib-2",
            "title": "Hotel booking refund exception and tight flight transfer",
            "messages": [
                {
                    "id": "c2-m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "I booked a non-refundable hotel room for a vacation next month, but an unexpected family medical situation came up. Is there any way to request an exception refund?"
                },
                {
                    "id": "c2-m2",
                    "parent_id": "c2-m1",
                    "role": "assistant",
                    "text": "Contact the hotel general manager or front office director directly by email rather than going through third-party call centers. Explain the unavoidable family emergency politely, provide relevant medical documentation if comfortable, and ask for a one-time refund exception or offer to convert the booking into a flexible date voucher for a future stay."
                },
                {
                    "id": "c2-m3",
                    "parent_id": "c2-m2",
                    "role": "user",
                    "text": "Should we confirm and submit the non-refundable hotel booking exception request we drafted?"
                },
                {
                    "id": "c2-m4",
                    "parent_id": "c2-m3",
                    "role": "assistant",
                    "text": "Yes, submit the exception form as drafted."
                }
            ]
        },

        # Calibration 3: Phone backup storage vs Bluetooth audio stutter
        {
            "source_id": "chatgpt",
            "account_id": "default",
            "id": "conv-calib-3",
            "title": "Cloud phone storage backup and wireless earbuds audio cutout",
            "messages": [
                {
                    "id": "c3-m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "My phone cloud storage alert says my cloud backup is completely full and stopping sync. How should I free up space without losing personal files?"
                },
                {
                    "id": "c3-m2",
                    "parent_id": "c3-m1",
                    "role": "assistant",
                    "text": "Review your cloud backup settings and turn off automatic backups for large streaming video apps, podcast caches, and redundant messaging attachments. Keep photos synced while offloading full-resolution video archives to a home computer or external drive, and delete obsolete device backups from old phones no longer in use."
                },
                {
                    "id": "c3-m3",
                    "parent_id": "c3-m2",
                    "role": "user",
                    "text": "My wireless earbuds keep stuttering and dropping audio connection for a second whenever I walk past busy street intersections. What causes this?"
                },
                {
                    "id": "c3-m4",
                    "parent_id": "c3-m3",
                    "role": "assistant",
                    "text": "Busy urban intersections are crowded with high-power twenty-four hundred megahertz radio frequency interference from traffic sensors, security cameras, and nearby cellular equipment. In your smartphone Bluetooth settings, switch the audio codec priority from maximum sound quality to stable connection mode, and keep your phone in an upper coat pocket to reduce body absorption."
                }
            ]
        },

        # Calibration 4: Laptop memory/performance (>1200 codepoints + astral) vs charging/battery
        {
            "source_id": "chatgpt",
            "account_id": "default",
            "id": "conv-calib-4",
            "title": "Multitasking laptop memory sluggishness and battery health threshold",
            "messages": [
                {
                    "id": "c4-m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "My laptop has been running noticeably slow and sluggish lately when I have several apps and browser tabs open at once. How can I free up memory and get performance back?"
                },
                {
                    "id": "c4-m2",
                    "parent_id": "c4-m1",
                    "role": "assistant",
                    "text": (
                        "Computing systems have evolved dramatically since the era of vacuum tubes and magnetic core storage. "
                        "Early mainframe computers occupied entire rooms and consumed kilowatts of power just to execute a few "
                        "thousand arithmetic instructions per second. Over the following decades, semiconductor miniaturization "
                        "led by silicon photolithography packed billions of nanometer transistors onto monolithic dies, completely "
                        "revolutionizing consumer electronics and digital signal processing architectures worldwide.\n\n"
                        "Modern operating systems rely on complex virtual memory managers that abstract physical silicon addresses "
                        "into contiguous address spaces. Disk storage architectures historically favored spinning platter hard drives "
                        "with mechanical arms and read-write heads, where rotational latency and seek times dictated throughput. "
                        "Today, solid-state drives using non-volatile flash memory and fast peripheral component interconnect express "
                        "lanes have virtually eliminated random access latency, changing how operating systems manage paging files "
                        "and swap partitions under peak utilization.\n\n"
                        "Beyond internal scheduling algorithms, ambient environmental conditions also influence computing reliability. "
                        "Thermal throttling safeguards sensitive silicon packages when heatsinks collect dust, reducing clock frequencies "
                        "to prevent semiconductor degradation. Cosmic radiation occasionally flips bits in unprotected volatile memories "
                        "at high altitudes, though error-correcting circuitry mitigates transient computational anomalies in high-reliability "
                        "server farms.\n\n"
                        "🚀🌟💻🔬⚡️\n\n"
                        "To immediately recover responsiveness and eliminate memory pressure on your laptop, inspect startup background "
                        "processes and disable unneeded launch daemons that load automatically on system boot. Modern web browsers create "
                        "separate isolated processes for every active tab and extension; close unused background extensions and enable memory "
                        "saver tab-discarding so background tabs release their physical memory allocations back to the operating system."
                    )
                },
                {
                    "id": "c4-m3",
                    "parent_id": "c4-m2",
                    "role": "user",
                    "text": "Should I leave my laptop plugged into the power charger all day while working at my desk, or does that ruin battery life?"
                },
                {
                    "id": "c4-m4",
                    "parent_id": "c4-m3",
                    "role": "assistant",
                    "text": "Modern laptops use intelligent power management chips that switch entirely to AC wall power once the battery reaches full charge, preventing continuous overcharging. However, keeping lithium-ion cells sitting at one hundred percent state of charge under warm operating temperatures accelerates long-term capacity degradation. Enable your laptop battery health management threshold to stop charging at eighty percent when primarily using desk AC power."
                }
            ]
        },

        # Calibration 5: Balcony plant watering vs pet-safe fertilizer
        {
            "source_id": "chatgpt",
            "account_id": "default",
            "id": "conv-calib-5",
            "title": "Balcony herb container watering and feline pet-safe plant fertilizer",
            "messages": [
                {
                    "id": "c5-m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "How can I tell when my balcony potted herbs and tomato plants actually need watering without risking root rot from overwatering?"
                },
                {
                    "id": "c5-m2",
                    "parent_id": "c5-m1",
                    "role": "assistant",
                    "text": "Check moisture with the two-inch finger test by inserting your index finger into the potting mix up to the middle knuckle; water thoroughly only when the soil feels completely dry below the surface. Always ensure container drainage holes remain unobstructed so excess water drains freely into drip trays, and water early in the morning so plant foliage dries before evening."
                },
                {
                    "id": "c5-m3",
                    "parent_id": "c5-m2",
                    "role": "user",
                    "text": "What are some safe natural fertilizer options for indoor potted houseplants that will not harm an inquisitive house cat?"
                },
                {
                    "id": "c5-m4",
                    "parent_id": "c5-m3",
                    "role": "assistant",
                    "text": "Avoid organic fertilizers containing blood meal, bone meal, or cocoa shell mulch, as their scent attracts curious cats and can cause acute toxicity or intestinal blockage if ingested. Use cold-water processed liquid kelp extract or mild fish hydrolysate diluted to half strength, or apply well-rotted worm castings directly to the soil surface for slow-release nutrients."
                }
            ]
        },

        # Calibration 6: Bicycle vs bus commute decision with regenerated sibling branch (conflicting decisions)
        {
            "source_id": "chatgpt",
            "account_id": "default",
            "id": "conv-calib-6",
            "title": "Commute decision between cycling and municipal bus with regenerated branch",
            "messages": [
                {
                    "id": "c6-m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "I need to choose my daily five-mile commute method between riding my bicycle and riding the city bus. Which option should I commit to?"
                },
                {
                    "id": "c6-m2-bike",
                    "parent_id": "c6-m1",
                    "role": "assistant",
                    "text": "Commit to the bicycle for your daily commute. Five miles is an ideal cycling distance that takes approximately twenty-five minutes while providing built-in cardiovascular exercise. Riding a bicycle gives you complete scheduling autonomy with zero timetable waiting, completely bypasses street congestion on the protected riverfront bike trail, and saves thousands of dollars annually on fares and parking fees."
                },
                {
                    "id": "c6-m2-bus",
                    "parent_id": "c6-m1",
                    "role": "assistant",
                    "text": "Commit to the city bus for your daily commute. Taking the municipal bus eliminates daily physical fatigue and provides a comfortable, climate-controlled space where you can read, study, or relax without navigating hazardous traffic. It keeps you clean and dry during rainy or freezing weather conditions, avoids dangerous bridge crossings during rush hour, and costs very little when using an employer-subsidized monthly transit pass."
                },
                {
                    "id": "c6-m3-bus",
                    "parent_id": "c6-m2-bus",
                    "role": "user",
                    "text": "What is the best way to manage fares and track bus arrival times for the daily bus commute?"
                },
                {
                    "id": "c6-m4-bus",
                    "parent_id": "c6-m3-bus",
                    "role": "assistant",
                    "text": "Download the municipal transit agency mobile app and purchase a digital thirty-day rolling bus pass linked to auto-reload. Enable real-time GPS vehicle tracking in the app to view live bus locations and set departure alert notifications ten minutes before your scheduled stop arrival time."
                }
            ]
        },

        # Calibration 7 (Chinese): Household storage cleanup vs phone photo backup
        {
            "source_id": "chatgpt",
            "account_id": "default",
            "id": "conv-calib-7",
            "title": "小户型换季衣物收纳整理与手机家庭照片双重备份",
            "messages": [
                {
                    "id": "c7-m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "换季时家里小户型衣柜和储物间堆放混乱，衣服和杂物越来越多，应该如何科学规划空间和分类整理？"
                },
                {
                    "id": "c7-m2",
                    "parent_id": "c7-m1",
                    "role": "assistant",
                    "text": "首先将所有衣物按季节进行集中断舍离，淘汰两年以上未穿的旧衣服。对于保留的厚重冬装，使用透明带拉链的收纳箱搭配真空压缩袋进行密封，防潮且节省三分之二空间。柜内隔板使用垂直竖立叠放法代替平铺堆叠，所有抽屉贴上标签注明类别，方便拿取且不易复乱。"
                },
                {
                    "id": "c7-m3",
                    "parent_id": "c7-m2",
                    "role": "user",
                    "text": "手机里积攒了上万张家庭生活照片和视频，系统频繁提示存储空间不足，怎样建立安全长效的备份机制？"
                },
                {
                    "id": "c7-m4",
                    "parent_id": "c7-m3",
                    "role": "assistant",
                    "text": "建立三级备份策略防止照片丢失。第一级在手机中按年份和重要事件创建归档相册，清理连拍和重复截图；第二级每月定期连接电脑，将原图完整拷贝至两块不同品牌的独立外置机械移动硬盘进行冷备份；第三级开启可信赖的云存储服务进行异地加密同步，确保多终端随时调阅与硬件防损。"
                }
            ]
        },

        # Calibration 8 (Chinese): Rice cooking texture vs grocery budget planning
        {
            "source_id": "chatgpt",
            "account_id": "default",
            "id": "conv-calib-8",
            "title": "煮大米饭口感水分调控与家庭食材买菜预算控制",
            "messages": [
                {
                    "id": "c8-m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "日常煮大米饭时米粒不是偏硬夹生就是发黏成团，不同米种的水米比例和淘洗浸泡方法有什么讲究？"
                },
                {
                    "id": "c8-m2",
                    "parent_id": "c8-m1",
                    "role": "assistant",
                    "text": "煮饭口感主要取决于米种含水量与淀粉结构。如果是东北长粒香米或五常稻花香，淘洗两遍去除表面浮粉后，加入冷水以体积比一比一点一至一比一点二为宜，提前在清水中浸泡二十分钟让米粒吸饱水分。电饭煲跳闸后不要立即揭盖，继续断电焖饭十分钟，再用饭勺从底部翻拌打散排出多余水汽，米饭才能粒粒油润分明。"
                },
                {
                    "id": "c8-m3",
                    "parent_id": "c8-m2",
                    "role": "user",
                    "text": "每月家庭去超市购买食材和生活日用品总是超出预算，有什么切实可行的计划采购和省钱方法？"
                },
                {
                    "id": "c8-m4",
                    "parent_id": "c8-m3",
                    "role": "assistant",
                    "text": "控制家庭饮食采购开支的核心在于提前规划与避免冲动消费。每周日对照冰箱现有存量制定周一至周五的固定荤素食谱，列出精准采购清单并严格按单购买。肉类选择大包装鲜肉，回家后按单次烹饪分量切块分装冷冻；蔬菜优先购买当季本地时令品种，坚决不空腹逛超市，即可在保证营养的同时降低三成伙食开销。"
                }
            ]
        },

        # Holdout 1: Rental security deposit dispute vs leaking tap repair
        {
            "source_id": "chatgpt",
            "account_id": "default",
            "id": "conv-hold-1",
            "title": "Rental security deposit wear and tear dispute and dripping sink tap fix",
            "messages": [
                {
                    "id": "h1-m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "Our former landlord is withholding half our rental security deposit for carpet wear and tear and wall scuffs after two years of tenancy. What are our rights?"
                },
                {
                    "id": "h1-m2",
                    "parent_id": "h1-m1",
                    "role": "assistant",
                    "text": "Residential tenancy regulations strictly prohibit landlords from deducting security deposits for ordinary wear and tear resulting from normal day-to-day living. Gather your move-in and move-out condition checklist photos, and send a formal certified demand letter requesting an itemized deduction ledger with repair invoices within statutory deadlines before filing a dispute in small claims court."
                },
                {
                    "id": "h1-m3",
                    "parent_id": "h1-m2",
                    "role": "user",
                    "text": "The bathroom sink faucet has developed a steady drip from the aerator spout even when both hot and cold handles are turned firmly shut. How do I fix it?"
                },
                {
                    "id": "h1-m4",
                    "parent_id": "h1-m3",
                    "role": "assistant",
                    "text": "A persistent faucet spout drip is almost always caused by a degraded rubber O-ring seat or a worn ceramic disc cartridge inside the handle assembly. Turn off the hot and cold water shutoff valves located directly underneath the vanity sink, pry off the decorative handle index caps, remove the retaining screws, and extract the old cartridge to match a direct replacement at the hardware store."
                }
            ]
        },

        # Holdout 2: Language pronunciation practice vs spaced repetition vocabulary
        {
            "source_id": "chatgpt",
            "account_id": "default",
            "id": "conv-hold-2",
            "title": "Spoken language pronunciation shadowing and spaced repetition review pacing",
            "messages": [
                {
                    "id": "h2-m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "I can read and understand foreign vocabulary fairly well, but my spoken pronunciation still sounds stiff and robotic when speaking full sentences. How can I improve my accent?"
                },
                {
                    "id": "h2-m2",
                    "parent_id": "h2-m1",
                    "role": "assistant",
                    "text": "Practice shadowing native audio material by listening through headphones and repeating full spoken sentences with a fraction of a second delay, mimicking pitch contour, word linking, and rhythm. Record short thirty-second audio clips on your phone to compare your natural cadence against native speaker waveforms, focusing on reducing unstressed vowel sounds rather than over-enunciating every syllable."
                },
                {
                    "id": "h2-m3",
                    "parent_id": "h2-m2",
                    "role": "user",
                    "text": "I want to use spaced repetition flashcards for vocabulary acquisition, but my daily review backlog keeps growing until I feel completely overwhelmed and quit. How should I pace it?"
                },
                {
                    "id": "h2-m4",
                    "parent_id": "h2-m3",
                    "role": "assistant",
                    "text": "Limit your daily intake to no more than ten brand-new vocabulary cards per day to keep total daily review obligations manageable and predictable. Always include full example sentences on cards to provide real grammatical context rather than memorizing isolated dictionary definitions, and complete your review queue during dedicated ten-minute morning routines before opening other apps."
                }
            ]
        },

        # Holdout 3 (Chinese): Chinese tea bitterness vs coffee pour-over flow
        {
            "source_id": "chatgpt",
            "account_id": "default",
            "id": "conv-hold-3",
            "title": "高山茶与绿茶冲泡苦涩味化解及手冲咖啡水流过萃排查",
            "messages": [
                {
                    "id": "h3-m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "平时冲泡高山乌龙茶和优质绿茶时，经常觉得茶汤苦涩难咽掩盖了花果香气，水温和出汤节奏该怎么把控？"
                },
                {
                    "id": "h3-m2",
                    "parent_id": "h3-m1",
                    "role": "assistant",
                    "text": "冲泡绿茶如龙井或碧螺春切忌直接用沸水，水烧开后应先倒入公道杯静置降温至八十到八十五摄氏度再注水，茶叶受热柔和才不会烫伤茶多酚释放涩味。前三泡均需在二十秒内快速出汤；而高山乌龙茶需用百度沸水激发出高扬香气，第一泡五秒润茶倒掉，第二泡十五秒出汤，之后每泡递增八秒，茶汤即可保持甘醇顺滑。"
                },
                {
                    "id": "h3-m3",
                    "parent_id": "h3-m2",
                    "role": "user",
                    "text": "在家手冲浅烘焙咖啡豆时，滴滤下水速度非常缓慢且后段萃取出的咖啡明显发苦发涩，应该如何排查原因？"
                },
                {
                    "id": "h3-m4",
                    "parent_id": "h3-m3",
                    "role": "assistant",
                    "text": "下水堵塞且发苦通常是研磨细粉过多或注水手法不当导致的过萃现象。首先检查磨豆机刻度，将研磨度调粗半格至白砂糖颗粒大小，筛除部分极细微粉以避免堵塞滤纸微孔。注水时保持水流平稳垂直且中心绕圈直径不超过硬币大小，总冲煮时间控制在两分钟到两分十五秒之间，避免水流长时间冲击滤纸边缘产生旁路死水。"
                }
            ]
        },

        # Holdout 4 (Chinese): Yunnan travel itinerary vs severe weather cancellation
        {
            "source_id": "chatgpt",
            "account_id": "default",
            "id": "conv-hold-4",
            "title": "陪伴父母云南慢节奏旅游行程规划与极端天气景区退款退费",
            "messages": [
                {
                    "id": "h4-m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "准备带年过六十的父母去云南做一次五天四晚的慢节奏旅行，怎样规划行程路线才能避开高反并避免长途舟车劳顿？"
                },
                {
                    "id": "h4-m2",
                    "parent_id": "h4-m1",
                    "role": "assistant",
                    "text": "带长辈出行应遵循低海拔逐步过渡和每天单一核心景点的原则。第一天飞抵昆明或大理低海拔地区入住休整，第二天乘动车前往大理古城慢逛洱海生态廊道，第三天转往丽江古城周边游览木府和束河古镇，第四天仅安排游览玉龙雪山脚下的蓝月谷避开主峰高反区域，第五天返程。全天行程包车点对点接送，午间保证两小时客栈午休，即可舒适尽兴。"
                },
                {
                    "id": "h4-m3",
                    "parent_id": "h4-m2",
                    "role": "user",
                    "text": "旅行途中如果遇到极端暴雨天气导致景区索道停运或班车取消，已提前预订的景区门票和包车费用怎么办理无损退款？"
                },
                {
                    "id": "h4-m4",
                    "parent_id": "h4-m3",
                    "role": "assistant",
                    "text": "遇到不可抗力导致景区关停时，第一时间用手机截取文旅部门或景区官方公众号发布的关停通告与天气预警信息作为凭证。通过官方购票平台提交不可抗力非自愿退款申请，系统审核后全额退回账户无需手续费。包车服务应立即与车队协商终止当天未执行行程，依据合同不可抗力条款退还未发生的路费与油费，必要时由当地文旅便民热线协助调解。"
                }
            ]
        }
    ]

    # Map conversations for lookup: (src, acc, cid) -> mid -> text, and parent links
    conv_lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    conv_parents: dict[tuple[str, str, str], dict[str, str | None]] = {}
    for c in conversations:
        k = (c.get("source_id", "chatgpt"), c.get("account_id", "default"), c["id"])
        conv_lookup[k] = {m["id"]: m["text"] for m in c["messages"]}
        conv_parents[k] = {m["id"]: m.get("parent_id") for m in c["messages"]}

    def get_ancestors(key: tuple[str, str, str], message_id: str) -> list[str]:
        ancestors = []
        cur_pid = conv_parents[key].get(message_id)
        seen = set()
        while cur_pid and cur_pid not in seen:
            seen.add(cur_pid)
            ancestors.append(cur_pid)
            cur_pid = conv_parents[key].get(cur_pid)
        return ancestors

    # Explicit 8 user-request-focused positive overrides for dataset role balance (8 user / 40 assistant)
    user_gold_overrides: dict[str, tuple[str, str, str]] = {
        "q-calib-pos-3": ("conv-calib-1", "c1-m3", "where did I describe dough refusing to rise in a cold kitchen"),
        "q-calib-pos-9": ("conv-calib-3", "c3-m1", "find my question about the phone backup stopping because online storage was full"),
        "q-calib-pos-15": ("conv-calib-4", "c4-m3", "the question I asked about leaving my laptop connected at the desk"),
        "q-calib-pos-17": ("conv-calib-5", "c5-m1", "when did I ask how to tell whether the balcony plants needed watering"),
        "q-calib-pos-25": ("conv-calib-7", "c7-m1", "我在哪里问过换季衣服太多，家里收纳地方不够的问题"),
        "q-hold-pos-1": ("conv-hold-1", "h1-m1", "my question about the landlord keeping the deposit over ordinary carpet wear"),
        "q-hold-pos-7": ("conv-hold-2", "h2-m3", "where did I complain that yesterday's vocabulary cards were piling up"),
        "q-hold-pos-13": ("conv-hold-4", "h4-m1", "我问过带父母旅游不想走太累的那条需求"),
    }

    raw_queries = [
        # --- Calibration Queries (32 Positives) ---
        # Conv 1: Exchange 1 (Tomato soup acidity)
        {
            "id": "q-calib-pos-1", "split": "calibration", "type": "positive",
            "query": "the cooking advice for making dinner taste less sharp without sweetening it",
            "golds": [("conv-calib-1", "c1-m2", ["c1-m1"],
                "Roasting the tomatoes in the oven caramelizes their natural sugars and softens harsh raw acidity before blending. When simmering the soup, grating in a fresh sweet carrot adds subtle sweetness, and stirring in a splash of heavy cream or a quarter teaspoon of baking soda neutralizes the sharp acidic taste without masking the tomato flavor.")]
        },
        {
            "id": "q-calib-pos-2", "split": "calibration", "type": "positive",
            "query": "oven roasting tomatoes and adding baking soda or dairy for soup acidity",
            "golds": [("conv-calib-1", "c1-m2", ["c1-m1"],
                "Roasting the tomatoes in the oven caramelizes their natural sugars and softens harsh raw acidity before blending. When simmering the soup, grating in a fresh sweet carrot adds subtle sweetness, and stirring in a splash of heavy cream or a quarter teaspoon of baking soda neutralizes the sharp acidic taste without masking the tomato flavor.")]
        },
        # Conv 1: Exchange 2 (Bread proofing user question)
        {
            "id": "q-calib-pos-3", "split": "calibration", "type": "positive",
            "query": "where did I describe dough refusing to rise in a cold kitchen",
            "golds": [("conv-calib-1", "c1-m3", ["c1-m2", "c1-m1"],
                "My bread dough takes forever to rise during winter because our kitchen is cold and drafty. How can I speed up proofing safely?")]
        },
        {
            "id": "q-calib-pos-4", "split": "calibration", "type": "positive",
            "query": "using unheated oven interior light as a warm dough proofing box",
            "golds": [("conv-calib-1", "c1-m4", ["c1-m3", "c1-m2", "c1-m1"],
                "To create an ideal warm fermentation environment without buying a dedicated proofing box, place your covered dough bowl inside an unheated oven with only the interior oven light turned on. The low-wattage incandescent bulb gently warms the sealed oven cavity to approximately eighty degrees Fahrenheit, providing ideal ambient temperature and consistent humidity for steady dough expansion.")]
        },

        # Conv 2: Exchange 1 (Hotel refund exception)
        {
            "id": "q-calib-pos-5", "split": "calibration", "type": "positive",
            "query": "the exception we discussed for a prepaid room after a family emergency",
            "golds": [("conv-calib-2", "c2-m2", ["c2-m1"],
                "Contact the hotel general manager or front office director directly by email rather than going through third-party call centers. Explain the unavoidable family emergency politely, provide relevant medical documentation if comfortable, and ask for a one-time refund exception or offer to convert the booking into a flexible date voucher for a future stay.")]
        },
        {
            "id": "q-calib-pos-6", "split": "calibration", "type": "positive",
            "query": "family medical emergency hotel cancellation voucher request",
            "golds": [("conv-calib-2", "c2-m2", ["c2-m1"],
                "Contact the hotel general manager or front office director directly by email rather than going through third-party call centers. Explain the unavoidable family emergency politely, provide relevant medical documentation if comfortable, and ask for a one-time refund exception or offer to convert the booking into a flexible date voucher for a future stay.")]
        },
        # Conv 2: Exchange 2 (Approval acknowledgement)
        {
            "id": "q-calib-pos-7", "split": "calibration", "type": "positive",
            "query": "where did you give the go-ahead to send the accommodation request",
            "golds": [("conv-calib-2", "c2-m4", ["c2-m3", "c2-m2", "c2-m1"],
                "Yes, submit the exception form as drafted.")]
        },
        {
            "id": "q-calib-pos-8", "split": "calibration", "type": "positive",
            "query": "confirming whether to send the nonrefundable room exception form now",
            "golds": [("conv-calib-2", "c2-m4", ["c2-m3", "c2-m2", "c2-m1"],
                "Yes, submit the exception form as drafted.")]
        },

        # Conv 3: Exchange 1 (Phone backup storage user question)
        {
            "id": "q-calib-pos-9", "split": "calibration", "type": "positive",
            "query": "find my question about the phone backup stopping because online storage was full",
            "golds": [("conv-calib-3", "c3-m1", [],
                "My phone cloud storage alert says my cloud backup is completely full and stopping sync. How should I free up space without losing personal files?")]
        },
        {
            "id": "q-calib-pos-10", "split": "calibration", "type": "positive",
            "query": "disabling app video backups and deleting old phone cloud archives",
            "golds": [("conv-calib-3", "c3-m2", ["c3-m1"],
                "Review your cloud backup settings and turn off automatic backups for large streaming video apps, podcast caches, and redundant messaging attachments. Keep photos synced while offloading full-resolution video archives to a home computer or external drive, and delete obsolete device backups from old phones no longer in use.")]
        },
        # Conv 3: Exchange 2 (Bluetooth audio stutter)
        {
            "id": "q-calib-pos-11", "split": "calibration", "type": "positive",
            "query": "the explanation for why my earpieces briefly go silent while walking outside",
            "golds": [("conv-calib-3", "c3-m4", ["c3-m3", "c3-m2", "c3-m1"],
                "Busy urban intersections are crowded with high-power twenty-four hundred megahertz radio frequency interference from traffic sensors, security cameras, and nearby cellular equipment. In your smartphone Bluetooth settings, switch the audio codec priority from maximum sound quality to stable connection mode, and keep your phone in an upper coat pocket to reduce body absorption.")]
        },
        {
            "id": "q-calib-pos-12", "split": "calibration", "type": "positive",
            "query": "switching bluetooth codec to stable connection mode to stop audio drops",
            "golds": [("conv-calib-3", "c3-m4", ["c3-m3", "c3-m2", "c3-m1"],
                "Busy urban intersections are crowded with high-power twenty-four hundred megahertz radio frequency interference from traffic sensors, security cameras, and nearby cellular equipment. In your smartphone Bluetooth settings, switch the audio codec priority from maximum sound quality to stable connection mode, and keep your phone in an upper coat pocket to reduce body absorption.")]
        },

        # Conv 4: Exchange 1 (Laptop memory pressure >1200 codepoints + astral)
        {
            "id": "q-calib-pos-13", "split": "calibration", "type": "positive",
            "query": "what we agreed would make my computer responsive with many things open",
            "golds": [("conv-calib-4", "c4-m2", ["c4-m1"],
                "To immediately recover responsiveness and eliminate memory pressure on your laptop, inspect startup background processes and disable unneeded launch daemons that load automatically on system boot. Modern web browsers create separate isolated processes for every active tab and extension; close unused background extensions and enable memory saver tab-discarding so background tabs release their physical memory allocations back to the operating system.")]
        },
        {
            "id": "q-calib-pos-14", "split": "calibration", "type": "positive",
            "query": "disabling startup launch daemons and enabling browser tab memory saver",
            "golds": [("conv-calib-4", "c4-m2", ["c4-m1"],
                "To immediately recover responsiveness and eliminate memory pressure on your laptop, inspect startup background processes and disable unneeded launch daemons that load automatically on system boot. Modern web browsers create separate isolated processes for every active tab and extension; close unused background extensions and enable memory saver tab-discarding so background tabs release their physical memory allocations back to the operating system.")]
        },
        # Conv 4: Exchange 2 (Laptop battery charging user question)
        {
            "id": "q-calib-pos-15", "split": "calibration", "type": "positive",
            "query": "the question I asked about leaving my laptop connected at the desk",
            "golds": [("conv-calib-4", "c4-m3", ["c4-m2", "c4-m1"],
                "Should I leave my laptop plugged into the power charger all day while working at my desk, or does that ruin battery life?")]
        },
        {
            "id": "q-calib-pos-16", "split": "calibration", "type": "positive",
            "query": "does keeping laptop at 100 percent charge degrade lithium ion capacity",
            "golds": [("conv-calib-4", "c4-m4", ["c4-m3", "c4-m2", "c4-m1"],
                "Modern laptops use intelligent power management chips that switch entirely to AC wall power once the battery reaches full charge, preventing continuous overcharging. However, keeping lithium-ion cells sitting at one hundred percent state of charge under warm operating temperatures accelerates long-term capacity degradation. Enable your laptop battery health management threshold to stop charging at eighty percent when primarily using desk AC power.")]
        },

        # Conv 5: Exchange 1 (Balcony plant watering user question)
        {
            "id": "q-calib-pos-17", "split": "calibration", "type": "positive",
            "query": "when did I ask how to tell whether the balcony plants needed watering",
            "golds": [("conv-calib-5", "c5-m1", [],
                "How can I tell when my balcony potted herbs and tomato plants actually need watering without risking root rot from overwatering?")]
        },
        {
            "id": "q-calib-pos-18", "split": "calibration", "type": "positive",
            "query": "preventing potted plant root rot and checking container drainage holes",
            "golds": [("conv-calib-5", "c5-m2", ["c5-m1"],
                "Check moisture with the two-inch finger test by inserting your index finger into the potting mix up to the middle knuckle; water thoroughly only when the soil feels completely dry below the surface. Always ensure container drainage holes remain unobstructed so excess water drains freely into drip trays, and water early in the morning so plant foliage dries before evening.")]
        },
        # Conv 5: Exchange 2 (Pet safe plant fertilizer)
        {
            "id": "q-calib-pos-19", "split": "calibration", "type": "positive",
            "query": "the plant-feeding options we discussed with a curious cat around",
            "golds": [("conv-calib-5", "c5-m4", ["c5-m3", "c5-m2", "c5-m1"],
                "Avoid organic fertilizers containing blood meal, bone meal, or cocoa shell mulch, as their scent attracts curious cats and can cause acute toxicity or intestinal blockage if ingested. Use cold-water processed liquid kelp extract or mild fish hydrolysate diluted to half strength, or apply well-rotted worm castings directly to the soil surface for slow-release nutrients.")]
        },
        {
            "id": "q-calib-pos-20", "split": "calibration", "type": "positive",
            "query": "using liquid kelp extract and worm castings for feline safe plant food",
            "golds": [("conv-calib-5", "c5-m4", ["c5-m3", "c5-m2", "c5-m1"],
                "Avoid organic fertilizers containing blood meal, bone meal, or cocoa shell mulch, as their scent attracts curious cats and can cause acute toxicity or intestinal blockage if ingested. Use cold-water processed liquid kelp extract or mild fish hydrolysate diluted to half strength, or apply well-rotted worm castings directly to the soil surface for slow-release nutrients.")]
        },

        # Conv 6: Exchange 1 (Bicycle recommendation branch c6-m2-bike)
        {
            "id": "q-calib-pos-21", "split": "calibration", "type": "positive",
            "query": "the greener commute option we considered along the protected riverside path",
            "golds": [("conv-calib-6", "c6-m2-bike", ["c6-m1"],
                "Commit to the bicycle for your daily commute. Five miles is an ideal cycling distance that takes approximately twenty-five minutes while providing built-in cardiovascular exercise. Riding a bicycle gives you complete scheduling autonomy with zero timetable waiting, completely bypasses street congestion on the protected riverfront bike trail, and saves thousands of dollars annually on fares and parking fees.")]
        },
        # Conv 6: Exchange 1 (Bus recommendation conflicting sibling branch c6-m2-bus)
        {
            "id": "q-calib-pos-22", "split": "calibration", "type": "positive",
            "query": "reasons to take city bus to stay dry in rain and avoid bridge traffic",
            "golds": [("conv-calib-6", "c6-m2-bus", ["c6-m1"],
                "Commit to the city bus for your daily commute. Taking the municipal bus eliminates daily physical fatigue and provides a comfortable, climate-controlled space where you can read, study, or relax without navigating hazardous traffic. It keeps you clean and dry during rainy or freezing weather conditions, avoids dangerous bridge crossings during rush hour, and costs very little when using an employer-subsidized monthly transit pass.")]
        },
        # Conv 6: Exchange 2 (Bus app and transit pass under c6-m2-bus)
        {
            "id": "q-calib-pos-23", "split": "calibration", "type": "positive",
            "query": "how to avoid standing at the bus stop guessing when it arrives",
            "golds": [("conv-calib-6", "c6-m4-bus", ["c6-m3-bus", "c6-m2-bus", "c6-m1"],
                "Download the municipal transit agency mobile app and purchase a digital thirty-day rolling bus pass linked to auto-reload. Enable real-time GPS vehicle tracking in the app to view live bus locations and set departure alert notifications ten minutes before your scheduled stop arrival time.")]
        },
        {
            "id": "q-calib-pos-24", "split": "calibration", "type": "positive",
            "query": "tracking municipal bus arrival with real time gps transit alerts",
            "golds": [("conv-calib-6", "c6-m4-bus", ["c6-m3-bus", "c6-m2-bus", "c6-m1"],
                "Download the municipal transit agency mobile app and purchase a digital thirty-day rolling bus pass linked to auto-reload. Enable real-time GPS vehicle tracking in the app to view live bus locations and set departure alert notifications ten minutes before your scheduled stop arrival time.")]
        },

        # Conv 7: Exchange 1 (Chinese storage cleanup user question)
        {
            "id": "q-calib-pos-25", "split": "calibration", "type": "positive",
            "query": "我在哪里问过换季衣服太多，家里收纳地方不够的问题",
            "golds": [("conv-calib-7", "c7-m1", [],
                "换季时家里小户型衣柜和储物间堆放混乱，衣服和杂物越来越多，应该如何科学规划空间和分类整理？")]
        },
        {
            "id": "q-calib-pos-26", "split": "calibration", "type": "positive",
            "query": "小衣柜抽屉垂直竖立折叠衣服贴标签分类规划",
            "golds": [("conv-calib-7", "c7-m2", ["c7-m1"],
                "首先将所有衣物按季节进行集中断舍离，淘汰两年以上未穿的旧衣服。对于保留的厚重冬装，使用透明带拉链的收纳箱搭配真空压缩袋进行密封，防潮且节省三分之二空间。柜内隔板使用垂直竖立叠放法代替平铺堆叠，所有抽屉贴上标签注明类别，方便拿取且不易复乱。")]
        },
        # Conv 7: Exchange 2 (Chinese photo backup)
        {
            "id": "q-calib-pos-27", "split": "calibration", "type": "positive",
            "query": "手机里的家庭照片太多又舍不得删，那次说怎么另存保险一点",
            "golds": [("conv-calib-7", "c7-m4", ["c7-m3", "c7-m2", "c7-m1"],
                "建立三级备份策略防止照片丢失。第一级在手机中按年份和重要事件创建归档相册，清理连拍和重复截图；第二级每月定期连接电脑，将原图完整拷贝至两块不同品牌的独立外置机械移动硬盘进行冷备份；第三级开启可信赖的云存储服务进行异地加密同步，确保多终端随时调阅与硬件防损。")]
        },
        {
            "id": "q-calib-pos-28", "split": "calibration", "type": "positive",
            "query": "两块独立机械硬盘备份手机家庭生活原图照片防损坏",
            "golds": [("conv-calib-7", "c7-m4", ["c7-m3", "c7-m2", "c7-m1"],
                "建立三级备份策略防止照片丢失。第一级在手机中按年份和重要事件创建归档相册，清理连拍和重复截图；第二级每月定期连接电脑，将原图完整拷贝至两块不同品牌的独立外置机械移动硬盘进行冷备份；第三级开启可信赖的云存储服务进行异地加密同步，确保多终端随时调阅与硬件防损。")]
        },

        # Conv 8: Exchange 1 (Chinese rice texture)
        {
            "id": "q-calib-pos-29", "split": "calibration", "type": "positive",
            "query": "上次饭煮出来黏成一团，建议我怎样调整",
            "golds": [("conv-calib-8", "c8-m2", ["c8-m1"],
                "煮饭口感主要取决于米种含水量与淀粉结构。如果是东北长粒香米或五常稻花香，淘洗两遍去除表面浮粉后，加入冷水以体积比一比一点一至一比一点二为宜，提前在清水中浸泡二十分钟让米粒吸饱水分。电饭煲跳闸后不要立即揭盖，继续断电焖饭十分钟，再用饭勺从底部翻拌打散排出多余水汽，米饭才能粒粒油润分明。")]
        },
        {
            "id": "q-calib-pos-30", "split": "calibration", "type": "positive",
            "query": "电饭煲煮大米饭跳闸后焖十分钟翻拌散水汽不发粘",
            "golds": [("conv-calib-8", "c8-m2", ["c8-m1"],
                "煮饭口感主要取决于米种含水量与淀粉结构。如果是东北长粒香米或五常稻花香，淘洗两遍去除表面浮粉后，加入冷水以体积比一比一点一至一比一点二为宜，提前在清水中浸泡二十分钟让米粒吸饱水分。电饭煲跳闸后不要立即揭盖，继续断电焖饭十分钟，再用饭勺从底部翻拌打散排出多余水汽，米饭才能粒粒油润分明。")]
        },
        # Conv 8: Exchange 2 (Chinese grocery budget)
        {
            "id": "q-calib-pos-31", "split": "calibration", "type": "positive",
            "query": "我嫌每周吃饭花钱太多时，我们定了什么采购办法",
            "golds": [("conv-calib-8", "c8-m4", ["c8-m3", "c8-m2", "c8-m1"],
                "控制家庭饮食采购开支的核心在于提前规划与避免冲动消费。每周日对照冰箱现有存量制定周一至周五的固定荤素食谱，列出精准采购清单并严格按单购买。肉类选择大包装鲜肉，回家后按单次烹饪分量切块分装冷冻；蔬菜优先购买当季本地时令品种，坚决不空腹逛超市，即可在保证营养的同时降低三成伙食开销。")]
        },
        {
            "id": "q-calib-pos-32", "split": "calibration", "type": "positive",
            "query": "大包装生鲜肉类分装冷冻与购买当季蔬菜省钱习惯",
            "golds": [("conv-calib-8", "c8-m4", ["c8-m3", "c8-m2", "c8-m1"],
                "控制家庭饮食采购开支的核心在于提前规划与避免冲动消费。每周日对照冰箱现有存量制定周一至周五的固定荤素食谱，列出精准采购清单并严格按单购买。肉类选择大包装鲜肉，回家后按单次烹饪分量切块分装冷冻；蔬菜优先购买当季本地时令品种，坚决不空腹逛超市，即可在保证营养的同时降低三成伙食开销。")]
        },

        # --- Holdout Queries (16 Positives, entirely disjoint conversations) ---
        # Holdout 1: Exchange 1 (Rental deposit dispute user question)
        {
            "id": "q-hold-pos-1", "split": "holdout", "type": "positive",
            "query": "my question about the landlord keeping the deposit over ordinary carpet wear",
            "golds": [("conv-hold-1", "h1-m1", [],
                "Our former landlord is withholding half our rental security deposit for carpet wear and tear and wall scuffs after two years of tenancy. What are our rights?")]
        },
        {
            "id": "q-hold-pos-2", "split": "holdout", "type": "positive",
            "query": "tenancy move out photo evidence disputing normal living carpet wear",
            "golds": [("conv-hold-1", "h1-m2", ["h1-m1"],
                "Residential tenancy regulations strictly prohibit landlords from deducting security deposits for ordinary wear and tear resulting from normal day-to-day living. Gather your move-in and move-out condition checklist photos, and send a formal certified demand letter requesting an itemized deduction ledger with repair invoices within statutory deadlines before filing a dispute in small claims court.")]
        },
        # Holdout 1: Exchange 2 (Leaking tap repair)
        {
            "id": "q-hold-pos-3", "split": "holdout", "type": "positive",
            "query": "the repair steps for water still trickling into the basin after turning it tight",
            "golds": [("conv-hold-1", "h1-m4", ["h1-m3", "h1-m2", "h1-m1"],
                "A persistent faucet spout drip is almost always caused by a degraded rubber O-ring seat or a worn ceramic disc cartridge inside the handle assembly. Turn off the hot and cold water shutoff valves located directly underneath the vanity sink, pry off the decorative handle index caps, remove the retaining screws, and extract the old cartridge to match a direct replacement at the hardware store.")]
        },
        {
            "id": "q-hold-pos-4", "split": "holdout", "type": "positive",
            "query": "turning off under sink shutoff valves to replace dripping faucet cartridge",
            "golds": [("conv-hold-1", "h1-m4", ["h1-m3", "h1-m2", "h1-m1"],
                "A persistent faucet spout drip is almost always caused by a degraded rubber O-ring seat or a worn ceramic disc cartridge inside the handle assembly. Turn off the hot and cold water shutoff valves located directly underneath the vanity sink, pry off the decorative handle index caps, remove the retaining screws, and extract the old cartridge to match a direct replacement at the hardware store.")]
        },

        # Holdout 2: Exchange 1 (Language pronunciation)
        {
            "id": "q-hold-pos-5", "split": "holdout", "type": "positive",
            "query": "the exercise we picked to stop guessing where English words should be stressed",
            "golds": [("conv-hold-2", "h2-m2", ["h2-m1"],
                "Practice shadowing native audio material by listening through headphones and repeating full spoken sentences with a fraction of a second delay, mimicking pitch contour, word linking, and rhythm. Record short thirty-second audio clips on your phone to compare your natural cadence against native speaker waveforms, focusing on reducing unstressed vowel sounds rather than over-enunciating every syllable.")]
        },
        {
            "id": "q-hold-pos-6", "split": "holdout", "type": "positive",
            "query": "recording phone audio to practice word linking and unstressed vowels",
            "golds": [("conv-hold-2", "h2-m2", ["h2-m1"],
                "Practice shadowing native audio material by listening through headphones and repeating full spoken sentences with a fraction of a second delay, mimicking pitch contour, word linking, and rhythm. Record short thirty-second audio clips on your phone to compare your natural cadence against native speaker waveforms, focusing on reducing unstressed vowel sounds rather than over-enunciating every syllable.")]
        },
        # Holdout 2: Exchange 2 (Spaced repetition pacing user question)
        {
            "id": "q-hold-pos-7", "split": "holdout", "type": "positive",
            "query": "where did I complain that yesterday's vocabulary cards were piling up",
            "golds": [("conv-hold-2", "h2-m3", ["h2-m2", "h2-m1"],
                "I want to use spaced repetition flashcards for vocabulary acquisition, but my daily review backlog keeps growing until I feel completely overwhelmed and quit. How should I pace it?")]
        },
        {
            "id": "q-hold-pos-8", "split": "holdout", "type": "positive",
            "query": "learning vocabulary with full sentence context rather than dictionary definitions",
            "golds": [("conv-hold-2", "h2-m4", ["h2-m3", "h2-m2", "h2-m1"],
                "Limit your daily intake to no more than ten brand-new vocabulary cards per day to keep total daily review obligations manageable and predictable. Always include full example sentences on cards to provide real grammatical context rather than memorizing isolated dictionary definitions, and complete your review queue during dedicated ten-minute morning routines before opening other apps.")]
        },

        # Holdout 3: Exchange 1 (Chinese tea bitterness)
        {
            "id": "q-hold-pos-9", "split": "holdout", "type": "positive",
            "query": "what we said to do when green tea comes out tasting unpleasantly dry and bitter",
            "golds": [("conv-hold-3", "h3-m2", ["h3-m1"],
                "冲泡绿茶如龙井或碧螺春切忌直接用沸水，水烧开后应先倒入公道杯静置降温至八十到八十五摄氏度再注水，茶叶受热柔和才不会烫伤茶多酚释放涩味。前三泡均需在二十秒内快速出汤；而高山乌龙茶需用百度沸水激发出高扬香气，第一泡五秒润茶倒掉，第二泡十五秒出汤，之后每泡递增八秒，茶汤即可保持甘醇顺滑。")]
        },
        {
            "id": "q-hold-pos-10", "split": "holdout", "type": "positive",
            "query": "高山乌龙茶沸水冲泡首泡润茶后按秒递增出汤时间",
            "golds": [("conv-hold-3", "h3-m2", ["h3-m1"],
                "冲泡绿茶如龙井或碧螺春切忌直接用沸水，水烧开后应先倒入公道杯静置降温至八十到八十五摄氏度再注水，茶叶受热柔和才不会烫伤茶多酚释放涩味。前三泡均需在二十秒内快速出汤；而高山乌龙茶需用百度沸水激发出高扬香气，第一泡五秒润茶倒掉，第二泡十五秒出汤，之后每泡递增八秒，茶汤即可保持甘醇顺滑。")]
        },
        # Holdout 3: Exchange 2 (Chinese coffee flow)
        {
            "id": "q-hold-pos-11", "split": "holdout", "type": "positive",
            "query": "the troubleshooting checklist for home drip brew that stalls and drains too slowly",
            "golds": [("conv-hold-3", "h3-m4", ["h3-m3", "h3-m2", "h3-m1"],
                "下水堵塞且发苦通常是研磨细粉过多或注水手法不当导致的过萃现象。首先检查磨豆机刻度，将研磨度调粗半格至白砂糖颗粒大小，筛除部分极细微粉以避免堵塞滤纸微孔。注水时保持水流平稳垂直且中心绕圈直径不超过硬币大小，总冲煮时间控制在两分钟到两分十五秒之间，避免水流长时间冲击滤纸边缘产生旁路死水。")]
        },
        {
            "id": "q-hold-pos-12", "split": "holdout", "type": "positive",
            "query": "浅烘咖啡手冲水流垂直中心绕圈控制两分钟过萃排查",
            "golds": [("conv-hold-3", "h3-m4", ["h3-m3", "h3-m2", "h3-m1"],
                "下水堵塞且发苦通常是研磨细粉过多或注水手法不当导致的过萃现象。首先检查磨豆机刻度，将研磨度调粗半格至白砂糖颗粒大小，筛除部分极细微粉以避免堵塞滤纸微孔。注水时保持水流平稳垂直且中心绕圈直径不超过硬币大小，总冲煮时间控制在两分钟到两分十五秒之间，避免水流长时间冲击滤纸边缘产生旁路死水。")]
        },

        # Holdout 4: Exchange 1 (Chinese Yunnan travel itinerary user question)
        {
            "id": "q-hold-pos-13", "split": "holdout", "type": "positive",
            "query": "我问过带父母旅游不想走太累的那条需求",
            "golds": [("conv-hold-4", "h4-m1", [],
                "准备带年过六十的父母去云南做一次五天四晚的慢节奏旅行，怎样规划行程路线才能避开高反并避免长途舟车劳顿？")]
        },
        {
            "id": "q-hold-pos-14", "split": "holdout", "type": "positive",
            "query": "老年人去云南旅游全天包车接送午间预留客栈午休",
            "golds": [("conv-hold-4", "h4-m2", ["h4-m1"],
                "带长辈出行应遵循低海拔逐步过渡和每天单一核心景点的原则。第一天飞抵昆明或大理低海拔地区入住休整，第二天乘动车前往大理古城慢逛洱海生态廊道，第三天转往丽江古城周边游览木府和束河古镇，第四天仅安排游览玉龙雪山脚下的蓝月谷避开主峰高反区域，第五天返程。全天行程包车点对点接送，午间保证两小时客栈午休，即可舒适尽兴。")]
        },
        # Holdout 4: Exchange 2 (Chinese severe weather cancellation)
        {
            "id": "q-hold-pos-15", "split": "holdout", "type": "positive",
            "query": "getting fees back without a penalty when weather shut down transport and attractions",
            "golds": [("conv-hold-4", "h4-m4", ["h4-m3", "h4-m2", "h4-m1"],
                "遇到不可抗力导致景区关停时，第一时间用手机截取文旅部门或景区官方公众号发布的关停通告与天气预警信息作为凭证。通过官方购票平台提交不可抗力非自愿退款申请，系统审核后全额退回账户无需手续费。包车服务应立即与车队协商终止当天未执行行程，依据合同不可抗力条款退还未发生的路费与油费，必要时由当地文旅便民热线协助调解。")]
        },
        {
            "id": "q-hold-pos-16", "split": "holdout", "type": "positive",
            "query": "不可抗力极端天气终止旅游包车合同退还未出行费用",
            "golds": [("conv-hold-4", "h4-m4", ["h4-m3", "h4-m2", "h4-m1"],
                "遇到不可抗力导致景区关停时，第一时间用手机截取文旅部门或景区官方公众号发布的关停通告与天气预警信息作为凭证。通过官方购票平台提交不可抗力非自愿退款申请，系统审核后全额退回账户无需手续费。包车服务应立即与车队协商终止当天未执行行程，依据合同不可抗力条款退还未发生的路费与油费，必要时由当地文旅便民热线协助调解。")]
        },

        # --- Unrelated Queries (16 total: 8 calibration, 8 holdout) ---
        # Near-domain plausible absent requests + disjoint topics
        {"id": "q-calib-unrel-1", "split": "calibration", "type": "unrelated", "query": "baking sourdough rye loaf with caraway seeds and dark molasses"},
        {"id": "q-calib-unrel-2", "split": "calibration", "type": "unrelated", "query": "troubleshooting microwave turntable glass roller guide motor replacement"},
        {"id": "q-calib-unrel-3", "split": "calibration", "type": "unrelated", "query": "winterizing swimming pool sand filter pump and chlorinator"},
        {"id": "q-calib-unrel-4", "split": "calibration", "type": "unrelated", "query": "使用高压锅炖牛腩白萝卜汤去腥软烂技巧"},
        {"id": "q-calib-unrel-5", "split": "calibration", "type": "unrelated", "query": "quantum entanglement bell inequality violation experimental tests"},
        {"id": "q-calib-unrel-6", "split": "calibration", "type": "unrelated", "query": "deep sea hydrothermal vent ecosystem hydrothermal mineral chimney"},
        {"id": "q-calib-unrel-7", "split": "calibration", "type": "unrelated", "query": "ancient Roman pozzolanic volcanic ash seawater concrete durability"},
        {"id": "q-calib-unrel-8", "split": "calibration", "type": "unrelated", "query": "Mars rover autonomous stereo vision obstacle hazard avoidance"},

        {"id": "q-hold-unrel-1", "split": "holdout", "type": "unrelated", "query": "修复老式机械自动上链手表发条摆轮走时偷停问题"},
        {"id": "q-hold-unrel-2", "split": "holdout", "type": "unrelated", "query": "申请欧洲申根签证在职证明与银行流水余额盖章要求"},
        {"id": "q-hold-unrel-3", "split": "holdout", "type": "unrelated", "query": "satellite low earth orbit inclination atmospheric aerodynamic drag"},
        {"id": "q-hold-unrel-4", "split": "holdout", "type": "unrelated", "query": "binary search tree left right red black node rotation rules"},
        {"id": "q-hold-unrel-5", "split": "holdout", "type": "unrelated", "query": "installing underground sprinkler drip irrigation timer for lawn"},
        {"id": "q-hold-unrel-6", "split": "holdout", "type": "unrelated", "query": "claiming compensation for cancelled cruise ship shore excursion"},
        {"id": "q-hold-unrel-7", "split": "holdout", "type": "unrelated", "query": "clearing cache memory on smart television streaming application"},
        {"id": "q-hold-unrel-8", "split": "holdout", "type": "unrelated", "query": "夏天挑选成熟红瓤黑籽甜西瓜拍打听声音技巧"}
    ]

    # Materialize gold offsets deterministically by exact unique expected_text lookup
    queries = []
    for item in raw_queries:
        qid = item["id"]
        q_obj = {
            "id": qid,
            "split": item["split"],
            "type": item["type"],
            "query": item["query"]
        }
        if "filters" in item:
            q_obj["filters"] = item["filters"]

        # Apply explicit user gold override with mechanical ancestry and bounds derivation
        if qid in user_gold_overrides:
            cid, mid, qtext = user_gold_overrides[qid]
            q_obj["query"] = qtext
            key = ("chatgpt", "default", cid)
            msg_text = conv_lookup[key][mid]
            ancestors = get_ancestors(key, mid)
            q_obj["expected_passages"] = [{
                "source_id": "chatgpt",
                "account_id": "default",
                "conversation_id": cid,
                "message_id": mid,
                "start_offset": 0,
                "end_offset": len(msg_text),
                "expected_text": msg_text,
                "expected_ancestor_ids": ancestors
            }]
            queries.append(q_obj)
            continue

        if item["type"] == "positive":
            expected_passages = []
            for g in item["golds"]:
                cid, mid, ancestors, exp_text = g
                key = ("chatgpt", "default", cid)
                if key not in conv_lookup or mid not in conv_lookup[key]:
                    raise RuntimeError(f"Unknown message {mid} in {cid} for query {item['id']}")
                msg_text = conv_lookup[key][mid]
                count = msg_text.count(exp_text)
                if count != 1:
                    raise RuntimeError(
                        f"expected_text must occur exactly once in {mid} for query {item['id']}, found {count}: {exp_text!r}"
                    )
                start_offset = msg_text.index(exp_text)
                end_offset = start_offset + len(exp_text)
                assert msg_text[start_offset:end_offset] == exp_text

                gold_entry = {
                    "source_id": "chatgpt",
                    "account_id": "default",
                    "conversation_id": cid,
                    "message_id": mid,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "expected_text": exp_text,
                    "expected_ancestor_ids": ancestors
                }
                expected_passages.append(gold_entry)
            q_obj["expected_passages"] = expected_passages

        queries.append(q_obj)

    suite = {
        "synthetic": True,
        "version": "passage-evaluation-v3",
        "description": (
            "Synthetic passage-level evaluation suite with ordinary-life ChatGPT topics, "
            "adversarial distractors, deep astral Unicode, multilingual dialogue, conflicting "
            "sibling branches, and entirely disjoint calibration/holdout splits."
        ),
        "conversations": conversations,
        "queries": queries
    }

    # Verify structural and contract constraints before serialization
    validate_suite(suite)
    return suite


def main() -> int:
    suite = build_suite()
    out_path = ROOT / "fixtures" / "passage-evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(suite, indent=2, ensure_ascii=False) + "\n"
    out_path.write_text(serialized, encoding="utf-8")
    print(f"Successfully generated and validated {out_path} ({len(suite['conversations'])} convs, {len(suite['queries'])} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
