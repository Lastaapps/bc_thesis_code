After I discovered how stupid I was I started using cycles that actually exist.


---------------------------------------------------------------------------------------------------------------------------- benchmark: 44 tests ----------------------------------------------------------------------------------------------------------------------------
Name (time in s)                                                                                                         Min                Max               Mean            StdDev             Median               IQR            Outliers     OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-none-8-smart]                                  4.5855 (1.0)       4.6159 (1.0)       4.6012 (1.0)      0.0144 (1.0)       4.6017 (1.0)      0.0242 (1.0)           2;0  0.2173 (1.0)           4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-none-6-smart]                                  4.9318 (1.08)      5.2876 (1.15)      5.0514 (1.10)     0.1649 (11.41)     4.9932 (1.09)     0.2264 (9.34)          1;0  0.1980 (0.91)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-neighbors-8-smart]                      4.9483 (1.0)       5.8985 (1.0)       5.2467 (1.0)      0.4419 (20.44)     5.0699 (1.0)      0.5436 (14.58)         1;0  0.1906 (1.0)           4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-neighbors_cycle-6-smart]                       5.1123 (1.04)      5.7005 (1.02)      5.2911 (1.0)      0.2746 (2.82)      5.1759 (1.02)     0.3011 (2.59)          1;0  0.1890 (1.0)           4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-neighbors_cycle-8-smart]                5.1636 (1.05)      5.5737 (1.0)       5.3297 (1.01)     0.1738 (1.78)      5.2908 (1.04)     0.2210 (1.90)          1;0  0.1876 (0.99)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-neighbors_cycle-8-smart]                       4.9221 (1.0)       6.3803 (1.14)      5.3616 (1.01)     0.6910 (7.09)      5.0720 (1.0)      0.8594 (7.38)          1;0  0.1865 (0.99)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-neighbors-6-smart]                             5.0811 (1.03)      6.0322 (1.02)      5.5866 (1.06)     0.4113 (19.02)     5.6165 (1.11)     0.6362 (17.07)         2;0  0.1790 (0.94)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-neighbors_cycle-6-smart]                5.6209 (1.14)      5.8555 (1.05)      5.6915 (1.08)     0.1114 (1.14)      5.6449 (1.11)     0.1397 (1.20)          1;0  0.1757 (0.93)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-neighbors-6-smart]                      5.5336 (1.12)      6.4371 (1.09)      5.8838 (1.12)     0.4064 (18.80)     5.7823 (1.14)     0.6034 (16.19)         1;0  0.1700 (0.89)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-neighbors_cycle-4-smart]                       5.7477 (1.17)      5.9735 (1.07)      5.8880 (1.11)     0.0975 (1.0)       5.9154 (1.17)     0.1164 (1.0)           1;0  0.1698 (0.90)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-neighbors_cycle-8-smart]                5.4320 (1.10)      6.6066 (1.12)      5.9080 (1.13)     0.5248 (24.27)     5.7966 (1.14)     0.7957 (21.34)         1;0  0.1693 (0.89)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-none-4-smart]                                  5.8361 (1.0)       6.0295 (1.0)       5.9450 (1.0)      0.0919 (2.83)      5.9573 (1.0)      0.1517 (2.94)          1;0  0.1682 (1.0)           4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-neighbors-8-smart]                             5.0604 (1.02)      6.7533 (1.14)      5.9826 (1.14)     0.7680 (35.52)     6.0584 (1.19)     1.2425 (33.33)         2;0  0.1672 (0.88)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-none-4-smart]                                  6.0876 (1.23)      6.1276 (1.04)      6.1064 (1.16)     0.0216 (1.0)       6.1052 (1.20)     0.0373 (1.0)           0;0  0.1638 (0.86)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-neighbors_iterative_cycle-8-smart]             5.3476 (1.09)      7.1163 (1.28)      6.1137 (1.16)     0.7436 (7.63)      5.9954 (1.18)     1.0232 (8.79)          2;0  0.1636 (0.87)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-neighbors_iterative_cycle-8-smart]      5.3116 (1.08)      7.6984 (1.38)      6.4041 (1.21)     1.1083 (11.37)     6.3032 (1.24)     1.8242 (15.68)         1;0  0.1562 (0.83)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-neighbors_iterative_cycle-6-smart]      5.6229 (1.14)      7.7595 (1.39)      6.4339 (1.22)     1.0258 (10.52)     6.1767 (1.22)     1.6203 (13.92)         1;0  0.1554 (0.82)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-neighbors_cycle-4-smart]                6.4023 (1.30)      6.8131 (1.22)      6.6256 (1.25)     0.1691 (1.73)      6.6434 (1.31)     0.2125 (1.83)          2;0  0.1509 (0.80)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-cycles-8-smart]                                6.7804 (1.37)      6.8579 (1.16)      6.8161 (1.30)     0.0373 (1.73)      6.8131 (1.34)     0.0627 (1.68)          1;0  0.1467 (0.77)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-cycles_match_chunks-6-smart]            6.5327 (1.32)      7.2376 (1.23)      6.8628 (1.31)     0.3391 (15.68)     6.8405 (1.35)     0.5699 (15.29)         1;0  0.1457 (0.76)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-cycles-6-smart]                         6.6921 (1.35)      7.5003 (1.27)      7.0285 (1.34)     0.4024 (18.61)     6.9608 (1.37)     0.6695 (17.96)         1;0  0.1423 (0.75)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-neighbors-4-smart]                             6.6234 (1.34)      7.9652 (1.35)      7.3215 (1.40)     0.5770 (26.69)     7.3487 (1.45)     0.8897 (23.87)         2;0  0.1366 (0.72)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-cycles-6-smart]                                6.7041 (1.35)      7.9302 (1.34)      7.3485 (1.40)     0.5992 (27.71)     7.3799 (1.46)     1.0140 (27.20)         1;0  0.1361 (0.71)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-cycles-8-smart]                         7.1561 (1.45)      7.5849 (1.29)      7.3601 (1.40)     0.2325 (10.75)     7.3497 (1.45)     0.4011 (10.76)         0;0  0.1359 (0.71)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-neighbors-4-smart]                      6.3679 (1.29)      8.7012 (1.48)      7.3614 (1.40)     1.0453 (48.35)     7.1884 (1.42)     1.6336 (43.82)         1;0  0.1358 (0.71)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-cycles_match_chunks-6-smart]                   7.1697 (1.45)      7.8318 (1.33)      7.4845 (1.43)     0.2711 (12.54)     7.4683 (1.47)     0.3428 (9.20)          2;0  0.1336 (0.70)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-cycles_match_chunks-8-smart]            7.4813 (1.51)      8.0522 (1.37)      7.7576 (1.48)     0.3096 (14.32)     7.7484 (1.53)     0.5346 (14.34)         0;0  0.1289 (0.68)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-neighbors_cycle-6-smart]                       6.3939 (1.29)     10.1045 (1.71)      7.7886 (1.48)     1.7752 (82.11)     7.3281 (1.45)     2.7829 (74.65)         1;0  0.1284 (0.67)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-neighbors_iterative_cycle-4-smart]      6.6071 (1.34)      9.5795 (1.72)      7.8790 (1.49)     1.2680 (13.00)     7.6647 (1.51)     1.8196 (15.64)         2;0  0.1269 (0.67)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-neighbors_iterative_cycle-4-smart]             7.3108 (1.49)      8.8140 (1.58)      7.9048 (1.49)     0.6545 (6.71)      7.7472 (1.53)     0.9192 (7.90)          1;0  0.1265 (0.67)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-cycles_match_chunks-8-smart]                   7.0742 (1.43)      9.1807 (1.56)      7.9299 (1.51)     0.9113 (42.15)     7.7323 (1.53)     1.2947 (34.73)         1;0  0.1261 (0.66)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-neighbors_cycle-6-smart]                7.8525 (1.59)      8.5839 (1.46)      8.1650 (1.56)     0.3434 (15.88)     8.1119 (1.60)     0.5592 (15.00)         1;0  0.1225 (0.64)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-neighbors_cycle-8-smart]                       5.8366 (1.18)     13.4733 (2.28)      8.2217 (1.57)     3.5821 (165.68)    6.7884 (1.34)     4.5661 (122.49)        1;0  0.1216 (0.64)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-neighbors_iterative_cycle-6-smart]             5.8803 (1.19)     11.8932 (2.13)      8.0189 (1.52)     2.7781 (28.49)     7.1511 (1.41)     4.0221 (34.56)         1;0  0.1247 (0.66)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-cycles_match_chunks-4-smart]                   8.0907 (1.39)      8.1625 (1.35)      8.1313 (1.37)     0.0325 (1.0)       8.1360 (1.37)     0.0516 (1.0)           1;0  0.1230 (0.73)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-neighbors_cycle-4-smart]                       8.1527 (1.40)      8.9815 (1.49)      8.5492 (1.44)     0.3988 (12.28)     8.5313 (1.43)     0.6716 (13.00)         1;0  0.1170 (0.70)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-none-subgraphs-linear-cycles-4-smart]                                8.3646 (1.69)      9.4061 (1.59)      8.8692 (1.69)     0.4777 (22.10)     8.8530 (1.75)     0.7865 (21.10)         2;0  0.1127 (0.59)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-neighbors_cycle-4-smart]                8.2942 (1.42)     10.6553 (1.77)      9.0252 (1.52)     1.1006 (33.89)     8.5757 (1.44)     1.3187 (25.53)         1;0  0.1108 (0.66)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-cycles-4-smart]                         9.4084 (1.90)     10.4377 (1.77)      9.9437 (1.90)     0.4721 (21.84)     9.9644 (1.97)     0.7765 (20.83)         2;0  0.1006 (0.53)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-cycles_match_chunks-4-smart]            8.6065 (1.47)     13.5704 (2.25)     10.2098 (1.72)     2.2975 (70.75)     9.3311 (1.57)     2.9442 (57.00)         1;0  0.0979 (0.58)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-none-6-smart]                          16.7874 (3.66)     19.7557 (4.28)     18.0455 (3.92)     1.2941 (89.56)    17.8194 (3.87)     1.9395 (80.01)         1;0  0.0554 (0.25)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-none-4-smart]                          17.0910 (3.45)     22.7009 (3.85)     18.9332 (3.61)     2.5557 (118.21)   17.9704 (3.54)     3.0799 (82.62)         1;0  0.0528 (0.28)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-none-4-smart]                          17.9391 (3.07)     20.5964 (3.42)     19.4306 (3.27)     1.3101 (40.34)    19.5934 (3.29)     2.1982 (42.56)         1;0  0.0515 (0.31)          4           1
test_bench_NAC_colorings_laman_fast[laman_larger-beam_degree-subgraphs-linear-none-8-smart]                          29.9963 (6.54)     63.5236 (13.76)    41.7004 (9.06)    15.0374 (>1000.0)  36.6409 (7.96)    19.4033 (800.49)        1;0  0.0240 (0.11)          4           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

















