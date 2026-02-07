// SPDX-License-Identifier: UNLICENSED

// pragma solidity ^0.7.4;
pragma solidity ^0.8.4;

contract RestricedAccess {
    address public owner = msg.sender;
    uint public creationTime = block.timestamp;

    modifier onlyBy(address _account) {
        require(msg.sender == _account, "Sender not Authorized!");
        _;
    }

    modifier onlyAfter(uint _time) {
        require(block.timestamp >= _time, "Function was called too early!");
        _;
    }

    modifier costs(uint _amount) {
        require(msg.value >= _amount, "Not enough ether provided!");
        _;
    }

    function forceChangeOwner(address _newOwner) payable public costs(50 ether) {
        owner = _newOwner;
    }

    function changeOwner(address _owner) public onlyBy(owner) {
        owner = _owner;
    }

    function disown() public onlyBy(owner) onlyAfter(creationTime + 5 seconds) {
        delete owner;
    }
}