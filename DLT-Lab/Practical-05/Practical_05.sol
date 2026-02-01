// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.7.4;

contract MyContract {
    address private owner;

    constructor() {
        owner = msg.sender;
    }

    function getOwner() public view returns (address) {
        return owner;
    }

    function getBalance() public view returns (uint256) {
        return owner.balance;
    }
}