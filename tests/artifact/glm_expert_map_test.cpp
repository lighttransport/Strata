// Explicit expert maps must fail closed before the packer reads or writes any model payload.
#include "strata/artifact/glm_expert_map.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char** argv) { try {
    const std::string row = "3 287 5\n", second = "4 18 1\n";
    {
        std::istringstream in(row + second);
        auto map = strata::artifact::read_glm_expert_map(in, 2, 3, 4);
        if (map.layers.at(3) != std::vector<int>{287, 5} || map.layers.at(4) != std::vector<int>{18, 1} ||
            map.plan_text != row + second) throw std::runtime_error("valid map changed its order or resume identity");
    }
    {
        std::istringstream in("# source ids\n\nexperts 288 2\n" + row);
        auto map = strata::artifact::read_glm_expert_map(in, 2, 3, 3);
        if (map.layers.at(3).at(0) != 287 || map.plan_text != row) throw std::runtime_error("plan header mismatch");
    }
    const std::vector<std::string> invalid = {
        "", "\n  \n", "not a mapping\n", "experts 288 2\n", row, second,
        row + second + row, row + "4 2 2\n", row + "4 -1 2\n", row + "4 288 2\n",
        row + "4 2\n", row + "4 2 3 4\n", row + "4 2 3 junk\n", row + "4 2 3x\n",
        "2 1 2\n" + row + second, row + second + "46 1 2\n",
        "experts 288 3\n" + row + second, "experts 144 2\n" + row + second,
        "experts 288 2 junk\n" + row + second, row + "experts 288 2\n" + second,
        "experts 288 2\nexperts 288 2\n" + row + second
    };
    for (const auto& text : invalid) {
        bool rejected = false;
        try { std::istringstream in(text); (void)strata::artifact::read_glm_expert_map(in, 2, 3, 4); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid map accepted");
    }
    if (argc == 2) {
        std::ifstream in(argv[1]);
        if (!in) throw std::runtime_error("cannot open shipped REAP plan");
        auto map = strata::artifact::read_glm_expert_map(in, 224, 3, 44);
        if (map.layers.size() != 43 || map.layers.at(3).at(223) != 287)
            throw std::runtime_error("shipped REAP plan differs");
    }
    std::cout << "expert map: valid order, plan compatibility and " << invalid.size() << " rejection cases passed\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
