// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.7.4;

contract LoopDemo {
    // 1. For Loop
    function forLoop(uint n) public pure returns (uint sum) {
        sum = 0;
        for(uint i = 1; i <= n; i++) {
            sum += i;
        }
    }

    // 2. While Loop
    function whileLoop(uint n) public pure returns (uint product) {
        product = 1;
        uint i = 1;
        while(i <= n) {
            product *= i;
            i++;
        }
    }

    // 3. Do-While Loop
    function doWhileLoop(uint n) public pure returns (uint count) {
        count = 0;
        uint i = 0;
        do {
            count++;
            i++;
        } while(i < n);
    }

    // 4. Nested Loop
    function nestedLoop(uint n) public pure returns (uint result) {
        result = 0;
        for(uint i = 0; i < n; i++) {
            for(uint j = 0; j < n; j++) {
                result += 1;
            }
        }
    }
}